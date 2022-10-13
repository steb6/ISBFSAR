import torch
from tqdm import tqdm
from modules.ar.utils.dataloader import FSOSEpisodicLoader, MyLoader
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig
import pickle
import os
from utils.params import ubuntu

query_path = os.path.join(".." if ubuntu else "D:", "datasets", "NTURGBD_to_YOLO_METRO")
exemplars_path = os.path.join(".." if ubuntu else "D:", "datasets", "NTURGBD_to_YOLO_METRO_exemplars")
checkpoints_path = "modules/ar/modules/raws/hybrid/2500.pth" if not ubuntu else "/home/IIT.LOCAL/sberti/ISBFSAR" \
                                                                                "/checkpoints/28_09_2022-09_17/2500.pth"

b_size = 1 if not ubuntu else 16
n_workers = 1 if not ubuntu else 8
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

    # GET MODEL
    model = TRXOS(args).cuda(device)
    state_dict = torch.load(checkpoints_path,
                            map_location=torch.device(0))["model_state_dict"]
    state_dict = {key.replace(".module", ""): state_dict[key] for key in state_dict.keys()}  # For DataParallel
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_grad_enabled(False)

    # For each test class
    for i, ss_name in enumerate(tqdm(test_classes)):
        # for i, ss_name in enumerate(test_classes):

        results[ss_name] = {}

        for q_name in test_classes:

            # Dataset Iterator
            test_data = MyLoader(query_path,
                                 exemplars_path=exemplars_path,
                                 support_classes=[ss_name],
                                 query_class=q_name)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=b_size, num_workers=1)

            os_preds = []
            targets = []
            count = 0
            # for elem in tqdm(test_loader):
            for elem in test_loader:
                # Extract from dict, convert, move to GPU
                support_set = [elem['support_set'][t].float().to(device) for t in elem['support_set'].keys()]
                target_set = [elem['target_set'][t].float().to(device) for t in elem['target_set'].keys()]

                support_labels = torch.arange(1).repeat(b_size).reshape(b_size, 1).to(device).int()
                target = torch.argmax((elem['support_classes'] == elem['target_class'][..., None]).int(),
                                      dim=1).float().to(device)

                with torch.no_grad():
                    out = model(support_set, support_labels, target_set)
                os_pred = out['is_true']

                os_preds.append(os_pred.reshape(-1))
                targets.append(elem["known"].cuda(device).reshape(-1))

                if count == 10:
                    break
                count += 1

            os_preds = torch.concat(os_preds) > 0.5
            targets = torch.concat(targets)

            score = (os_preds == targets).sum() / targets.numel()

            # good_indices = (targets == -1).nonzero(as_tuple=True)[0]
            # perc_true = (os_preds[good_indices].float().sum() / torch.numel(os_preds[good_indices])).item()

            results[ss_name][q_name] = score
            # TODO FILTER WHEN TARGETS IS 1

        print(results)

    with open("CONFUSIONMATRIX_2", "wb") as f:
        pickle.dump(results, f)
