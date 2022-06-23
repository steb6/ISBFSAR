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
    results = {}

    for model_type in ["DISC"]:  # , "EXP", "DISC-NO-OS"]:

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

        # For each K
        for ss_name in test_classes:

            results[ss_name] = {}

            for q_name in tqdm(test_classes, desc=f"{test_classes.index(ss_name)}"):

                # Dataset Iterator
                test_data = TestMetrabsData(args.data_path, "D:\\datasets\\metrabs_trx_skeletons_exemplars",
                                            [ss_name], [q_name])
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=1)

                os_preds = []
                targets = []

                for elem in test_loader:
                    support_set, target_set, _, support_labels, target_label, _, _, _ = elem
                    batch_size = support_set.size(0)

                    support_set = support_set.reshape(batch_size, 1, args.seq_len, args.n_joints * 3).cuda().float()
                    target_set = target_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
                    support_labels = support_labels.reshape(batch_size, 1).cuda().int()
                    target_label = target_label.cuda()

                    ################
                    # Known action #
                    ################
                    out = model(support_set, support_labels, target_set)
                    os_pred = out['is_true']

                    os_preds.append(os_pred)
                    targets.append(target_label)

                os_preds = torch.concat(os_preds) > 0.5
                targets = torch.concat(targets)
                good_indices = (targets == -1).nonzero(as_tuple=True)[0]
                perc_true = (os_preds[good_indices].float().sum() / torch.numel(os_preds[good_indices])).item()

                results[ss_name][q_name] = perc_true
                # TODO FILTER WHEN TARGETS IS 1

            print(results)

    with open("CONFUSIONMATRIX", "wb") as f:
        pickle.dump(results, f)

