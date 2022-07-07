import torch
from tqdm import tqdm
from modules.ar.utils.dataloader import TestMetrabsData
from modules.ar.utils.model import Skeleton_TRX_EXP, Skeleton_TRX_Disc
from utils.matplotlib_visualizer import MPLPosePrinter
from utils.params import TRXConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle

# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash

skeleton = 'smpl+head_30'
with open('assets/skeleton_types.pkl', "rb") as input_file:
    skeleton_types = pickle.load(input_file)
edges = skeleton_types[skeleton]['edges']

device = 0

# test_classes = ["A1",  "A19", "A25", "A37", "A49", "A67", "A79", "A91", "A97"]
# os_classes =   ["A7", "A13", "A31", "A43", "A61", "A73", "A85", "A103"]
os_classes = ["A7", "A31", "A61", "A85", "A103"]
test_classes = ["A1", "A13", "A19", "A25", "A37"]

aux = test_classes
test_classes = os_classes
os_classes = aux

test_way = len(test_classes)

# Get conversion class id -> class label
with open("assets/nturgbd_classes.txt", "r", encoding='utf-8') as f:
    classes = f.readlines()
class_dict = {}
for c in classes:
    index, name, _ = c.split(".")
    name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
    class_dict[index] = name
test_classes = [class_dict[elem] for elem in test_classes]
os_classes = [class_dict[elem] for elem in os_classes]

if __name__ == "__main__":
    args = TRXConfig()

    # Get right model
    trx_model = None
    if args.model == "DISC":
        trx_model = Skeleton_TRX_Disc
    elif args.model == "EXP":
        trx_model = Skeleton_TRX_EXP
    else:
        raise Exception("NOT a valid model")
    model = trx_model(args).cuda(device)
    model.load_state_dict(torch.load("modules/ar/modules/raws/DISC-NO-OS.pth", map_location=torch.device(0))["model_state_dict"])
    model.eval()

    # Create dataset iterator
    test_data = TestMetrabsData(args.data_path, "D:\\datasets\\metrabs_trx_skeletons_exemplars", test_classes, os_classes)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=args.n_workers)

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
    # vis = MPLPosePrinter()

    fs_preds = []
    os_preds = []
    targets = []

    for i, elem in enumerate(tqdm(test_loader)):
        support_set, target_set, _, support_labels, target_label = elem
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
        os_pred = out['is_true']

        # fs_pred = torch.rand(fs_pred.size()).cuda()  # TODO REMOVE RANDOMNESS
        # os_pred = torch.rand(os_pred.size()).cuda()  # TODO REMOVE RANDOMNESS

        fs_pred = torch.argmax(fs_pred, dim=1)
        is_true = os_pred > 0.5
        os_pred = torch.where(is_true.squeeze(-1), fs_pred, -1)
        # # TODO REMOVE DEBUG
        # for frame in target_set[0]:
        #     vis.set_title("TARGET")
        #     vis.clear()
        #     vis.print_pose(frame.detach().cpu().numpy().reshape(-1, 3), edges)
        #     vis.sleep(0.01)
        # for frame in support_set[0][0]:
        #     vis.set_title("SUPPORT")
        #     vis.clear()
        #     vis.print_pose(frame.detach().cpu().numpy().reshape(-1, 3), edges)
        #     vis.sleep(0.01)
        # # TODO END
        # # TODO CONTINUE
        fs_preds.append(fs_pred)
        os_preds.append(os_pred)
        targets.append(target_label)
    # WANDB
    fs_preds = torch.concat(fs_preds)
    os_preds = torch.concat(os_preds)
    targets = torch.concat(targets)

    acc = (os_preds == targets).float().mean().item()

    fs_indices = (targets != -1).nonzero(as_tuple=True)[0]
    fs_acc = (fs_preds[fs_indices] == targets[fs_indices]).float().mean().item()

    # os_indices = (targets == -1).nonzero(as_tuple=True)[0]

    preds = torch.where(os_preds == -1, 0, 1)
    targets = torch.where(targets == -1, 0, 1)

    os_acc = (preds == targets).float().mean().item()
    # os_f1 = f1_score(preds.cpu(), targets.cpu())
    # os_acc = accuracy_score(preds.cpu(), targets.cpu())
    # os_rec = recall_score(preds.cpu(), targets.cpu())
    # os_prec = precision_score(preds.cpu(), targets.cpu())

    print("ACC: ", acc)
    print("FS ACC: ", fs_acc)
    # print("OS ACC: ", os_acc)
    # print("f1: ", os_f1)
    print("acc: ", os_acc)
    # print("rec: ", os_rec)
    # print("prec: ", os_prec)
