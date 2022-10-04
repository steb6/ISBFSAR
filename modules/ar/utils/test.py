import torch
import wandb
from tqdm import tqdm
from modules.ar.utils.dataloader import EpisodicLoader
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
# EPOCH 300
# {'train/fs_loss': 0.87, 'train/fs_accuracy': 0.53, 'train/os_loss': 0.68, 'train/os_accuracy': 0.67, 'train/os_precision': 0.75, 'train/os_recall': 0.52, 'train/os_f1': 0.62, 'train/os_n_1_true': 0.5, 'train/os_n_1_pred': 0.35, 'os_outs': 0.49}
# EPOCH 583
# {'train/fs_loss': 0.55, 'train/fs_accuracy': 0.71, 'train/os_loss': 0.62, 'train/os_accuracy': 0.69, 'train/os_precision': 0.66, 'train/os_recall': 0.76, 'train/os_f1': 0.71, 'train/os_n_1_true': 0.5, 'train/os_n_1_pred': 0.57, 'os_outs': 0.45}
# EPOCH 967
# {'train/fs_loss': 0.72, 'train/fs_accuracy': 0.71, 'train/os_loss': 0.44, 'train/os_accuracy': 0.80, 'train/os_precision': 0.80, 'train/os_recall': 0.80, 'train/os_f1': 0.80, 'train/os_n_1_true': 0.5, 'train/os_n_1_pred': 0.5, 'os_outs': 0.49}
# EPOCH 1212
# {'train/fs_loss': 0.54, 'train/fs_accuracy': 0.78, 'train/os_loss': 0.50, 'train/os_accuracy': 0.80, 'train/os_precision': 0.76, 'train/os_recall': 0.86, 'train/os_f1': 0.81, 'train/os_n_1_true': 0.5, 'train/os_n_1_pred': 0.56, 'os_outs': 0.53}
# EPOCH 3000
# {'train/fs_loss': 0.4730205172672868, 'train/fs_accuracy': 0.875, 'train/os_loss': 1.0021080905642066, 'train/os_accuracy': 0.6851851851851852, 'train/os_precision': 0.6923076923076923, 'train/os_recall': 0.6666666666666666, 'train/os_f1': 0.6792452830188679, 'train/os_n_1_true': 0.5, 'train/os_n_1_pred': 0.4814815, 'os_outs': 0.4980449479859061}


device = TRXConfig().device

if __name__ == "__main__":
    args = TRXConfig()

    b = args.batch_size

    # Get conversion class id -> class label
    test_classes = ["A1", "A7", "A13", "A19", "A25", "A31", "A37", "A43", "A49", "A55", "A61", "A67", "A73", "A79",
                    "A85",
                    "A91", "A97", "A103", "A109", "A115"]
    with open("assets/nturgbd_classes.txt", "r", encoding='utf-8') as f:
        classes = f.readlines()
    class_dict = {}
    for c in classes:
        index, name, _ = c.split(".")
        name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
        class_dict[index] = name
    test_classes = [class_dict[elem] for elem in test_classes]

    # Get model
    model = TRXOS(args).to(device)
    state_dict = torch.load("modules/ar/modules/raws/hybrid/l8_res50_e3000.pth")["model_state_dict"]
    state_dict = {key.replace(".module", ""): state_dict[key] for key in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()

    # Create dataset iterator
    train_data = EpisodicLoader(args.data_path, k=args.way, n_task=args.n_task, input_type=args.input_type, )

    # Exclude test classes and create iterator
    train_data.classes = list(filter(lambda x: x in test_classes, train_data.classes))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers,
                                               shuffle=True)

    # Log
    print("Testing time :DDDD have you trained out enough?: {}".format(len(train_loader)))
    print("Training for {} epochs".format(args.n_epochs))
    print("Batch size is {}".format(args.batch_size))

    # Losses
    os_loss_fn = torch.nn.BCELoss()
    fs_loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        fs_train_losses = []
        fs_train_accuracies = []
        os_train_losses = []
        os_train_pred = []
        os_train_true = []
        os_outs = []

        # TRAIN
        model.train()
        for i, elem in enumerate(tqdm(train_loader)):

            # Extract from dict, convert, move to GPU
            support_set = [elem['support_set'][t].float().to(device) for t in elem['support_set'].keys()]
            target_set = [elem['target_set'][t].float().to(device) for t in elem['target_set'].keys()]
            unknown_set = [elem['unknown_set'][t].float().to(device) for t in elem['unknown_set'].keys()]

            support_labels = torch.arange(args.way).repeat(b).reshape(b, args.way).to(device).int()
            target = (elem['support_classes'] == elem['target_class'][..., None]).float().to(device)

            ################
            # Known action #
            ################
            out = model(support_set, support_labels, target_set)

            # FS known
            fs_pred = out['logits']
            known_fs_loss = fs_loss_fn(fs_pred, target)
            fs_train_losses.append(known_fs_loss.item())
            final_loss = known_fs_loss

            train_accuracy = torch.eq(torch.argmax(fs_pred, dim=1), torch.argmax(target, dim=1)).float().mean().item()
            fs_train_accuracies.append(train_accuracy)

            # OS known (target depends on true class)
            os_pred = out['is_true']
            os_outs.append(os_pred.detach().cpu().numpy())
            target = torch.eq(torch.argmax(fs_pred, dim=1), torch.argmax(target, dim=1)).float().unsqueeze(-1)
            # Train only on correct prediction
            true_s = (target == 1.).nonzero(as_tuple=True)[0]
            n = len(true_s)
            os_pred = os_pred[true_s]
            target = target[true_s]

            if target.sum() > 0:
                known_os_loss = os_loss_fn(os_pred, target)
                os_train_true.append(target.cpu().numpy())
                os_train_pred.append((os_pred > 0.5).float().cpu().numpy())
                os_train_losses.append(known_os_loss.item())

                ##################
                # Unknown action #
                ##################
                out = model(support_set, support_labels, unknown_set)

                # OS unknown
                os_pred = out['is_true']
                os_outs.append(os_pred.detach().cpu().numpy())
                target = torch.zeros_like(os_pred)
                # Get n samples
                os_pred = os_pred[:n]
                target = target[:n]

                unknown_os_loss = os_loss_fn(os_pred, target)
                os_train_losses.append(unknown_os_loss.item())
                os_train_true.append(target.cpu().numpy())
                os_train_pred.append((os_pred > 0.5).float().cpu().numpy())

        # WANDB
        os_train_true = np.concatenate(os_train_true, axis=None) if len(os_train_true) > 0 else np.zeros(1)
        os_train_pred = np.concatenate(os_train_pred, axis=None) if len(os_train_pred) > 0 else np.zeros(1)
        print({"train/fs_loss": sum(fs_train_losses) / len(fs_train_losses),
               "train/fs_accuracy": sum(fs_train_accuracies) / len(fs_train_accuracies),
               "train/os_loss": (sum(os_train_losses) / len(os_train_losses)) if len(os_train_losses) > 0 else 0,
               "train/os_accuracy": accuracy_score(os_train_true, os_train_pred),
               "train/os_precision": precision_score(os_train_true, os_train_pred, zero_division=0),
               "train/os_recall": recall_score(os_train_true, os_train_pred, zero_division=0),
               "train/os_f1": f1_score(os_train_true, os_train_pred, zero_division=0),
               "train/os_n_1_true": os_train_true.mean(),
               "train/os_n_1_pred": os_train_pred.mean(),
               "os_outs": sum(np.concatenate(os_outs, axis=0).reshape(-1))/len(np.concatenate(os_outs, axis=0).reshape(-1))})
