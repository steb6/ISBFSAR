import math
import torch
from torch import nn
from itertools import combinations
from torch.autograd import Variable
from modules.ar.trx.utils.misc import split_first_dim_linear

NUM_SAMPLES = 1


class PositionalEncoding(nn.Module):
    "Implement the PE function."

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
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
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

        self.class_softmax = torch.nn.Softmax(dim=1)

        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).to(args.device) for comb in frame_combinations]
        self.tuples_len = len(self.tuples)

    def forward(self, support_set, support_labels, queries):
        n_queries = 1  # queries.shape[0]
        n_support = 5  # support_set.shape[0]

        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]  # TODO PARAMETRIZE
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]  # TODO PARAMETRIZE
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
        for c in support_labels:
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, self.tuples_len)  # -1

            # TODO BEFORE
            # class_scores_real = self.class_softmax(class_scores[0])
            # TODO NEW
            max_along_axis = class_scores[0].max(dim=1, keepdim=True).values
            exponential = torch.exp(class_scores[0] - max_along_axis)
            denominator = torch.sum(exponential, dim=1, keepdim=True)
            denominator = denominator.repeat(1, self.tuples_len)
            class_scores = torch.div(exponential, denominator)
            # TODO END

            # class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, 1, self.tuples_len)  # -1
            class_scores = class_scores.permute(0, 2, 1, 3)

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)

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

        return_dict = {'logits': torch.stack(all_distances_tensor, dim=1)}

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


class CNN_TRX(nn.Module):
    """
    Standard Resnet connected to a Temporal Cross Transformer.

    """

    def __init__(self, args):
        super(CNN_TRX, self).__init__()

        self.train()
        self.args = args

        self.trans_linear_in_dim = args.trans_linear_in_dim
        self.features_extractor = MLP(self.args.n_joints * 3,
                                      self.args.n_joints * 3 * 2, self.trans_linear_in_dim)

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])

    def forward(self, context_images, context_labels, target_images):
        context_features = self.features_extractor(context_images).squeeze()
        target_features = self.features_extractor(target_images).squeeze()

        dim = self.trans_linear_in_dim

        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]])}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)