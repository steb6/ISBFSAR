import torch
from tqdm import tqdm
from modules.ar.utils.dataloader import FSOSEpisodicLoader
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig, ubuntu
import random
import pickle
import os
import numpy as np

# Multiply time with #CHECKPOINTS and #REPETITIONS: 10min x 7 x 5 => 350min => 6 hours

b = 3 if not ubuntu else 28
nw = 1 if not ubuntu else 16
device = 0
results = {}
query_path = os.path.join(".." if ubuntu else "D:", "datasets", "NTURGBD_to_YOLO_METRO")
exemplars_path = os.path.join(".." if ubuntu else "D:", "datasets", "NTURGBD_to_YOLO_METRO_exemplars")
checkpoints_path = "modules/ar/modules/raws/hybrid" if not ubuntu else "/home/IIT.LOCAL/sberti/ISBFSAR/checkpoints" \
                                                                       "/28_09_2022-09_17"

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

    for checkpoint in ["3000"]:  # , "EXP", "DISC-NO-OS"]:

        results[checkpoint] = {"FSOS-ACC": [],
                               "FS-ACC": [],
                               "OS-ACC": [],
                               # "OS-F1": []
                               }

        # GET MODEL
        model = TRXOS(args).cuda(device)
        state_dict = torch.load(os.path.join(checkpoints_path, "{}.pth".format(checkpoint)),
                                map_location=torch.device(0))["model_state_dict"]
        if False:  # ubuntu:
            model.distribute_model()
        else:
            state_dict = {key.replace(".module", ""): state_dict[key] for key in state_dict.keys()}  # For DataParallel
        model.load_state_dict(state_dict)
        model.eval()
        torch.set_grad_enabled(False)

        # Gor each K
        for K in [5]:  # list(range(5, 16)):  # Those are 12

            print("NEW K ################################")
            print("K:", K)

            results[checkpoint]["FSOS-ACC"].append([])
            results[checkpoint]["FS-ACC"].append([])
            results[checkpoint]["OS-ACC"].append([])
            # results[checkpoint]["OS-F1"].append([])

            for _ in range(10):  # Repeat
                # Dataset Iterator
                support_classes = random.sample(range(0, len(test_classes)), K)
                test_data = FSOSEpisodicLoader(query_path,
                                               exemplars_path,
                                               support_classes)
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=b, num_workers=nw,
                                                          persistent_workers=True)

                fs_score = []
                os_score = []
                fsos_score = []

                for elem in tqdm(test_loader):
                    # Extract from dict, convert, move to GPU
                    support_set = [elem['support_set'][t].float().to(device) for t in elem['support_set'].keys()]
                    target_set = [elem['target_set'][t].float().to(device) for t in elem['target_set'].keys()]

                    support_labels = torch.arange(args.way).repeat(b).reshape(b, args.way).to(device).int()
                    target = torch.argmax((elem['support_classes'] == elem['target_class'][..., None]).int(), dim=1).float().to(device)

                    with torch.no_grad():
                        out = model(support_set, support_labels, target_set)
                    fs_pred = out['logits']
                    os_pred = out['is_true']

                    # OS score depends only on itself
                    true_os = (os_pred > 0.5) == elem["known"].unsqueeze(-1).to(device)
                    os_score.append(true_os.detach().cpu().numpy())

                    # Compute true FS
                    fs_pred = torch.argmax(fs_pred, dim=1)
                    true_fs = fs_pred == target  # Note: results here could be no-sense (because of target)
                    true_fs_known = true_fs[elem["known"]]
                    fs_score.append(true_fs_known.detach().cpu().numpy())

                    # Compute FSOS
                    kn = torch.logical_and(elem["known"].to(device), true_fs)
                    kn = torch.logical_and(kn.unsqueeze(-1), true_os)
                    ukn = torch.logical_and(~elem["known"].to(device).unsqueeze(-1), true_os)
                    fsos_score.append(torch.logical_or(kn, ukn).detach().cpu().numpy())

                if len(fsos_score) > 0:
                    fsos_score = np.concatenate(fsos_score, axis=0).reshape(-1)
                    fsos_score = (fsos_score.sum() / fsos_score.size)
                else:
                    fsos_score = -1

                if len(fs_score) > 0:
                    fs_score = np.concatenate(fs_score, axis=0).reshape(-1)
                    fs_score = (fs_score.sum() / fs_score.size)
                else:
                    fs_score = -1

                if len(os_score) > 0:
                    os_score = np.concatenate(os_score, axis=0).reshape(-1)
                    os_score = (os_score.sum() / os_score.size)
                else:
                    os_score = -1
                    # f1_score = 0

                print("FSOS-ACC: ", fsos_score)
                print("FS ACC: ", fs_score)
                print("OS-ACC: ", os_score)
                # print("F1-SCORE: ", f1_score)

                results[checkpoint]["FSOS-ACC"][K - 5].append(fsos_score)
                results[checkpoint]["FS-ACC"][K - 5].append(fs_score)
                results[checkpoint]["OS-ACC"][K - 5].append(os_score)
                # results[checkpoint]["OS-F1"][K - 5].append(f1_score)

    with open("RESULTS_3000_5_REP", "wb") as f:
        pickle.dump(results, f)
