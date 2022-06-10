import torch
from tqdm import tqdm
from modules.ar.utils.dataloader import TestMetrabsData
from modules.ar.utils.misc import aggregate_accuracy, loss_fn
from modules.ar.utils.model import CNN_TRX
from utils.params import TRXConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash


test_way = 17
test_classes = ["A1", "A7", "A13", "A19", "A25", "A31", "A37", "A43", "A49", "A55", "A61", "A67", "A73", "A79", "A85",
                "A91", "A97", "A103", "A109", "A115"]
# Get conversion class id -> class label
with open("assets/nturgbd_classes.txt", "r", encoding='utf-8') as f:
    classes = f.readlines()
class_dict = {}
for c in classes:
    index, name, _ = c.split(".")
    name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
    class_dict[index] = name
test_classes = [class_dict[elem]for elem in test_classes]

if __name__ == "__main__":
    args = TRXConfig()

    model = CNN_TRX(args).cuda()
    model.load_state_dict(torch.load("modules/ar/213.pth", map_location=torch.device(0))["model_state_dict"])
    model.eval()

    # Create dataset iterator
    test_data = TestMetrabsData(args.data_path, "D:\\datasets\\nturgbd_trx_skeletons_exemplars", test_classes)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=args.n_workers)

    # Log
    print("Test samples: {}".format(len(test_loader)))

    # Open set loss
    os_loss_fn = torch.nn.BCEWithLogitsLoss()

    fs_valid_losses = []
    fs_valid_accuracies = []
    os_valid_losses = []
    os_valid_pred = []
    os_valid_true = []

    # EVAL
    model.eval()
    torch.set_grad_enabled(False)
    for i, elem in tqdm(enumerate(tqdm(test_loader)), position=0, leave=True):
        support_set, target_set, unknown_set, support_labels, target_label = elem

        support_set = support_set.reshape(test_way * args.seq_len, args.n_joints * 3).cuda().float()
        target_set = target_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
        support_labels = support_labels.reshape(test_way).cuda().int()
        target_label = target_label.cuda()

        ################
        # Known action #
        ################
        out = model(support_set, support_labels, target_set)

        # FS known
        # fs_pred = out['logits']
        fs_pred = torch.rand((1, 17)).cuda()

        target = torch.zeros_like(support_labels)
        target[target_label.item()] = 1.

        known_fs_loss = loss_fn(fs_pred.unsqueeze(0), target.unsqueeze(0), 'cuda')
        fs_valid_losses.append(known_fs_loss.item())

        valid_accuracy = aggregate_accuracy(fs_pred, target_label)
        fs_valid_accuracies.append(valid_accuracy.item())

        # OS known (target depends on true class)
        # os_pred = out['is_true']
        os_pred = torch.rand((1, 1)).cuda()
        target = 1 if torch.argmax(fs_pred) == target_label else 0
        known_os_loss = os_loss_fn(os_pred, torch.FloatTensor([target]).unsqueeze(0).cuda())
        os_valid_true.append(target)
        os_valid_pred.append(1 if os_pred.item() > 0.5 else 0)
        os_valid_losses.append(known_os_loss.item())

    print({"valid/fs_loss": sum(fs_valid_losses) / len(fs_valid_losses),
           "valid/fs_accuracy": sum(fs_valid_accuracies) / len(fs_valid_accuracies),
           "valid/os_loss": (sum(os_valid_losses) / len(os_valid_losses)) if len(os_valid_losses) > 0 else 0,
           "valid/os_accuracy": accuracy_score(os_valid_true, os_valid_pred),
           "valid/os_precision": precision_score(os_valid_true, os_valid_pred, zero_division=0),
           "valid/os_recall": recall_score(os_valid_true, os_valid_pred, zero_division=0),
           "valid/os_f1": f1_score(os_valid_true, os_valid_pred, zero_division=0)})
