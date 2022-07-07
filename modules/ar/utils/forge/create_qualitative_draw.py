import time

import torch
import os
from tqdm import tqdm
import random
from modules.ar.utils.dataloader import TestMetrabsData
from modules.ar.utils.model import Skeleton_TRX_EXP, Skeleton_TRX_Disc
from utils.matplotlib_visualizer import MPLPosePrinter
from utils.params import TRXConfig
import pickle

device = 0
results = {}
K = 3

# DEBUG
out_dir = "testing"
skeleton = 'smpl+head_30'
with open('assets/skeleton_types.pkl', "rb") as input_file:
    skeleton_types = pickle.load(input_file)
edges = skeleton_types[skeleton]['edges']

test_classes = ["A1", "A7", "A13", "A19", "A25", "A31", "A37", "A43", "A49",
                # "A55",
                "A61", "A67", "A73", "A79", "A85",
                "A91", "A97", "A103",
                # "A109", "A115"
                ]

if __name__ == "__main__":
    args = TRXConfig()

    # GET MODEL
    model_type = "DISC"
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

    # Get conversion class id -> class label
    with open("assets/nturgbd_classes.txt", "r", encoding='utf-8') as f:
        classes = f.readlines()
    class_dict = {}
    for c in classes:
        index, name, _ = c.split(".")
        name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
        class_dict[index] = name

    model.eval()
    model.cuda()
    torch.set_grad_enabled(False)

    vis = MPLPosePrinter()

    # Create dataset iterator
    for _ in range(1000):
        k_random_classes = random.sample(test_classes, K)
        others = [elem for elem in test_classes if elem not in k_random_classes]
        k_random_classes = [class_dict[elem] for elem in k_random_classes]
        others = [class_dict[elem] for elem in others]
        test_data = TestMetrabsData(args.data_path, "D:\\datasets\\metrabs_trx_skeletons_exemplars", k_random_classes, others)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=args.n_workers)

        for i, elem in enumerate(tqdm(test_loader)):
            support_set, query, unknown, support_labels, target_label, ss_names, trg_names, un_name = elem

            batch_size = support_set.size(0)
            # with open("assets/saved/support_set.pkl", "rb") as infile:
            #     support_set = pickle.load(infile)
            # with open("assets/saved/support_labels.pkl", "rb") as infile:
            #     support_name = pickle.load(infile)
            # query = support_set[3]
            # support_set = support_set[:3]
            # batch_size = 1

            support_set = support_set.reshape(batch_size, K, args.seq_len, args.n_joints * 3).cuda().float()
            query = query.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            unknown = unknown.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = support_labels.cuda()
            # unknown = unknown.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            # support_labels = torch.arange(0, 3).unsqueeze(0).cuda()

            out = model(support_set, support_labels, query)
            out_un = model(support_set, support_labels, unknown)
            fs_pred = torch.softmax(out['logits'], dim=1)
            fs_pred_un = torch.softmax(out_un['logits'], dim=1)

            # VISUALIZATION
            ss_names = [elem[0] for elem in ss_names]
            print(ss_names)
            print("target: ", trg_names[0])
            print("pred: ", fs_pred[0])
            print("is_true: ", out['is_true'][0])
            print("unknown class: ", un_name[0])
            print("pred: ", fs_pred_un[0])
            print("is_true: ", out_un['is_true'][0])

            # Visualize
            for i, sequence in enumerate(support_set[0]):
                time.sleep(1)
                vis.set_title("SUPPORT :"+ss_names[i])
                os.mkdir(os.path.join(out_dir, ss_names[i]))
                for j, pose in enumerate(sequence):
                    vis.clear()
                    vis.print_pose(pose.cpu().numpy().reshape(-1, 3), edges)
                    vis.save(os.path.join(out_dir, ss_names[i], str(j)))
                    vis.sleep(0.01)

            time.sleep(1)
            vis.set_title("QUERY :"+trg_names[0])
            os.mkdir(os.path.join(out_dir, "QUERY-"+trg_names[0]))
            for j, pose in enumerate(query[0]):
                vis.clear()
                vis.print_pose(pose.cpu().numpy().reshape(-1, 3), edges)
                vis.save(os.path.join(os.path.join(out_dir, "QUERY-"+trg_names[0]), str(j)))
                vis.sleep(0.01)

            time.sleep(1)
            vis.set_title("UNKNOWN :"+un_name[0])
            os.mkdir(os.path.join(out_dir, "UNKNOWN-"+un_name[0]))
            for j, pose in enumerate(unknown[0]):
                vis.clear()
                vis.print_pose(pose.cpu().numpy().reshape(-1, 3), edges)
                vis.save(os.path.join(os.path.join(out_dir, "UNKNOWN-"+un_name[0]), str(j)))
                vis.sleep(0.01)

            input()
            break
