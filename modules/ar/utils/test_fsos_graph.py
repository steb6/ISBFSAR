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
import random
import pickle

device = 0
results = {}

# DEBUG
skeleton = 'smpl+head_30'
with open('assets/skeleton_types.pkl', "rb") as input_file:
    skeleton_types = pickle.load(input_file)
edges = skeleton_types[skeleton]['edges']

# GET CLASSES
test_classes = ["A1", "A7", "A13", "A19", "A25", "A31", "A37", "A43", "A49", "A61", "A67", "A73", "A79", "A85", "A91",
                "A97", "A103"]
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

    for model_type in ["DISC", "EXP", "DISC-NO-OS"]:

        results[model_type] = {"FSOS-ACC": [],
                               "FS-ACC": [],
                               "OS-ACC": [],
                               "OS-F1": []}

        # GET MODEL
        trx_model = None
        if model_type == "DISC" or model_type == "DISC-NO-OS":
            trx_model = Skeleton_TRX_Disc
        elif model_type == "EXP":
            trx_model = Skeleton_TRX_EXP
        else:
            raise Exception("NOT a valid model")
        model = trx_model(args).cuda(device)
        model.load_state_dict(
            torch.load("modules/ar/modules/raws/{}.pth".format(model_type),
                       map_location=torch.device(0))["model_state_dict"])
        model.eval()
        torch.set_grad_enabled(False)

        # Gor each K
        for K in tqdm(list(range(5, 16))):

            results[model_type]["FSOS-ACC"].append([])
            results[model_type]["FS-ACC"].append([])
            results[model_type]["OS-ACC"].append([])
            results[model_type]["OS-F1"].append([])

            for _ in range(100):  # Repeat

                # Dataset Iterator
                support_classes = random.sample(range(0, len(test_classes)), K)
                support_classes = [test_classes[elem] for elem in support_classes]
                os_classes = [item for item in test_classes if item not in support_classes]
                test_data = TestMetrabsData(args.data_path, "D:\\datasets\\metrabs_trx_skeletons_exemplars",
                                            support_classes, os_classes)
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=12)

                # vis = MPLPosePrinter()

                fs_preds = []
                os_preds = []
                targets = []

                for elem in test_loader:
                    support_set, target_set, _, support_labels, target_label = elem
                    batch_size = support_set.size(0)

                    support_set = support_set.reshape(batch_size, K, args.seq_len, args.n_joints * 3).cuda().float()
                    target_set = target_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
                    support_labels = support_labels.reshape(batch_size, K).cuda().int()
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
                f1 = f1_score(preds.detach().cpu().numpy(), targets.detach().cpu().numpy())

                print("FSOS-ACC: ", acc)
                print("FS ACC: ", fs_acc)
                print("OS-ACC: ", os_acc)
                print("f1-scure: ", f1)

                results[model_type]["FSOS-ACC"][K-5].append(acc)
                results[model_type]["FS-ACC"][K-5].append(fs_acc)
                results[model_type]["OS-ACC"][K-5].append(os_acc)
                results[model_type]["OS-F1"][K - 5].append(f1)

    with open("RESULTS", "wb") as f:
        pickle.dump(results, f)
