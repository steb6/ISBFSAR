import torch
from tqdm import tqdm
from modules.ar.utils.dataloader import TestMetrabsData
import numpy as np
from modules.ar.utils.model import Skeleton_TRX_EXP, Skeleton_TRX_Disc
from utils.params import TRXConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash

model_type = "DISC"
test_way = 20
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
test_classes = [class_dict[elem] for elem in test_classes]

if __name__ == "__main__":
    args = TRXConfig()

    trx_model = None
    if model_type == "DISC":
        trx_model = Skeleton_TRX_Disc
    elif model_type == "EXP":
        trx_model = Skeleton_TRX_EXP
    else:
        raise Exception("NOT a valid model")
    model = trx_model(args)
    model.load_state_dict(torch.load("modules/ar/modules/raws/DISC-20_NOOS.pth", map_location=torch.device(0))["model_state_dict"])
    model.cuda()
    model.eval()

    # Create dataset iterator
    test_data = TestMetrabsData(args.data_path, "D:\\datasets\\nturgbd_trx_skeletons_exemplars", test_classes, [])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=12)

    # Open set loss
    os_loss_fn = torch.nn.BCELoss()
    fs_loss_fn = torch.nn.CrossEntropyLoss()

    fs_test_losses = []
    fs_test_accuracies = []
    os_test_losses = []
    os_test_pred = []
    os_test_true = []

    # TRAIN
    model.eval()
    torch.set_grad_enabled(False)
    for i, elem in enumerate(tqdm(test_loader)):
        support_set, target_set, _, support_labels, target_label, _, _, _ = elem
        batch_size = support_set.size(0)

        support_set = support_set.reshape(batch_size, test_way, args.seq_len, args.n_joints * 3).cuda().float()
        target_set = target_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
        # unknown_set = unknown_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
        support_labels = support_labels.reshape(batch_size, test_way).cuda().int()
        target_label = target_label.cuda()

        ################
        # Known action #
        ################
        out = model(support_set, support_labels, target_set)

        # FS known
        fs_pred = out['logits']
        target = torch.zeros_like(fs_pred)
        target[torch.arange(batch_size), target_label.long()] = 1
        scores = torch.softmax(fs_pred, dim=1)
        known_fs_loss = fs_loss_fn(scores, target)
        fs_test_losses.append(known_fs_loss.item())
        final_loss = known_fs_loss

        train_accuracy = torch.eq(torch.argmax(scores, dim=1), target_label).int().float()
        fs_test_accuracies.append(train_accuracy)

        # # OS known (target depends on true class)
        # os_pred = out['is_true']
        # target = torch.eq(torch.argmax(fs_pred, dim=1), target_label).float().unsqueeze(-1)
        # known_os_loss = os_loss_fn(os_pred, target)
        # os_test_true.append(target.cpu().numpy())
        # os_test_pred.append((os_pred > 0.5).float().cpu().numpy())
        # os_test_losses.append(known_os_loss.item())
    # WANDB
    # os_test_true = np.concatenate(os_test_true, axis=None)
    # os_test_pred = np.concatenate(os_test_pred, axis=None)
    fs_test_accuracies = torch.concat(fs_test_accuracies).reshape(-1).mean().item()

    print({"train/fs_loss": sum(fs_test_losses) / len(fs_test_losses),
           "train/fs_accuracy": fs_test_accuracies,
           "train/os_loss": (sum(os_test_losses) / len(os_test_losses)) if len(
               os_test_losses) > 0 else 0,
           # "train/os_accuracy": accuracy_score(os_test_true, os_test_pred),
           # "train/os_precision": precision_score(os_test_true, os_test_pred, zero_division=0),
           # "train/os_recall": recall_score(os_test_true, os_test_pred, zero_division=0),
           # "train/os_f1": f1_score(os_test_true, os_test_pred, zero_division=0)
    })
