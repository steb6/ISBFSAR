import math
import torch
from torch import nn
from itertools import combinations
from torch.autograd import Variable
from torchvision.models import resnet50, ResNet
from torchvision.models.resnet import Bottleneck

NUM_SAMPLES = 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(-2)], requires_grad=False)
        return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        # self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=-2)

        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).to(args.device) for comb in frame_combinations]
        self.tuples_len = len(self.tuples)

    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[1]
        n_support = support_set.shape[1]
        batch_size = queries.shape[0]

        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(batch_size, n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(batch_size, n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs

        # unique_labels = torch.unique(support_labels)  # TODO READD
        # I REMOVED THE LINE ABOVE BECAUSE UNIQUE IS NOT SUPPORTED IN TRT
        # unique_labels = support_labels

        # init tensor to hold distances between every support tuple and every target tuple
        # all_distances_tensor = torch.zeros(n_queries, self.args.way).cuda()
        all_distances_tensor = []
        diffs = []
        for c in support_labels[0]:
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, -3, c)
            class_v = torch.index_select(mh_support_set_vs, -3, c)
            # k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks, class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            # class_scores = class_scores.permute(0, 2, 1, 3)
            # class_scores = class_scores.reshape(batch_size, n_queries, self.tuples_len, self.tuples_len)  # -1

            # TODO BEFORE
            class_scores = self.class_softmax(class_scores)
            # TODO NEW
            # max_along_axis = class_scores.max(dim=-2, keepdim=True).values
            # exponential = torch.exp(class_scores[0] - max_along_axis)
            # denominator = torch.sum(exponential, dim=-2, keepdim=True)
            # denominator = denominator.repeat(1, self.tuples_len)
            # class_scores = torch.div(exponential, denominator)
            # TODO END

            # class_scores = torch.cat(class_scores)
            # class_scores = class_scores.reshape(batch_size, n_queries, self.tuples_len, 1, self.tuples_len)  # -1
            # class_scores = class_scores.permute(0, 1, 3, 2, 4)

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)
            # query_prototype = torch.sum(query_prototype, dim=1)

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
            distance = torch.div(norm_sq, self.tuples_len)

            # multiply by -1 to get logits
            distance = distance * -1
            all_distances_tensor.append(distance)
            # c_idx = c.long()
            # all_distances_tensor[:, c_idx] = 1  # distance
            # all_distances_tensor = all_distances_tensor + c_idx + distance

            diffs.append(diff)

        all_distances_tensor = torch.cat(all_distances_tensor, dim=1)

        return_dict = {'logits': all_distances_tensor, 'diffs': torch.concat(diffs, dim=1)}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        return which_class.reshape((-1,))
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu1(hidden)
        output_raw = self.fc2(relu)
        output = self.relu2(output_raw)
        return output


class Discriminator(torch.nn.Module):
    def __init__(self, seq_len=120, dim=128, l=16):
        super(Discriminator, self).__init__()
        self.dimensionality_reduction = torch.nn.Linear(dim, l)
        self.fc1 = torch.nn.Linear(seq_len*l, 256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 64)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        y = self.dimensionality_reduction(x)
        y = y.reshape(batch_size, -1)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)
        y = self.out(y)
        return y


class TRXOS(nn.Module):

    class myresnet50(ResNet):
        def __init__(self, pretrained=True):
            super().__init__(Bottleneck, [3, 4, 6, 3])
            self.gradients = None
            self.activations = []

        def activations_hook(self, grad):
            self.gradients = grad

        # method for the gradient extraction
        def get_activations_gradient(self):
            return self.gradients

        # method for the activation exctraction
        def get_activations(self):
            return self.activations

        def forward(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            h = x.register_hook(self.activations_hook)
            self.activations.append(x.detach())

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

    def __init__(self, args, add_hook=False):
        super(TRXOS, self).__init__()
        self.args = args

        self.trans_linear_in_dim = args.trans_linear_in_dim
        if args.input_type == "skeleton":
            self.features_extractor = MLP(args.n_joints * 3,
                                          args.n_joints * 3 * 2, self.trans_linear_in_dim)
        if args.input_type == "rgb":
            if add_hook:
                self.features_extractor = self.myresnet50(pretrained=True)
            else:
                self.features_extractor = resnet50(pretrained=True)

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])

        self.model = args.model
        if self.model == "DISC":
            self.discriminator = Discriminator(seq_len=int(((args.seq_len-1)*args.seq_len)/2),
                                               dim=args.trans_linear_out_dim,
                                               l=args.seq_len)
        if self.model == "EXP":
            self.discriminator = torch.exp

    def forward(self, context_images, context_labels, target_images):
        b = context_images.size(0)
        # with torch.no_grad():
        if self.args.input_type == "rgb":
            b, k, l, c, h, w = context_images.size()
            context_features = self.features_extractor(context_images.reshape(-1, c, h, w)).reshape(b, k, l, -1)
            target_features = self.features_extractor(target_images.reshape(-1, c, h, w)).reshape(b, l, -1)
        else:
            context_features = self.features_extractor(context_images)
            target_features = self.features_extractor(target_images)

        target_features = target_features.unsqueeze(1)
        out = self.transformers[0](context_features, context_labels, target_features)
        all_logits = out['logits']

        chosen_index = torch.argmax(all_logits, dim=1)
        feature = out['diffs'][torch.arange(b), chosen_index, ...]
        decision = self.discriminator(feature)

        return {'logits': all_logits, 'is_true': decision}

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.features_extractor.cuda(0)
            self.features_extractor = torch.nn.DataParallel(self.features_extractor, device_ids=[i for i in range(0, self.args.num_gpus)])
            self.features_extractor.cuda(0)


if __name__ == "__main__":
    from utils.params import TRXConfig
    model = TRXOS(TRXConfig()).cuda()

    model(torch.rand((1, 5, 16, 90)).cuda(),
          torch.randint(5, (1, 5)).cuda(),
          torch.rand((1, 16, 90)).cuda())
