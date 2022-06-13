import os
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
from modules.ar.utils.dataloader import MetrabsData
# from modules.ar.utils.misc import aggregate_accuracy, loss_fn
from modules.ar.utils.model import CNN_TRX
from utils.params import TRXConfig
import random
from utils.params import ubuntu
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

device = 0 if ubuntu else 0
torch.cuda.set_device(device)

# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash

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
    torch.manual_seed(0)

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    checkpoints_path = "checkpoints" + os.sep + datetime.now().strftime("%d_%m_%Y-%H_%M")
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    model = CNN_TRX(args).cuda(device)
    model.train()

    # Create dataset iterator
    train_data = MetrabsData(args.data_path, k=5, n_task=10000 if ubuntu else 100)
    valid_data = MetrabsData(args.data_path, k=5, n_task=1000 if ubuntu else 10)
    test_data = MetrabsData(args.data_path, k=5, n_task=1000)

    # Divide dataset into train and validation
    classes = train_data.classes
    inv_class_dict = {v: k for k, v in class_dict.items()}
    filtered_classes = []
    for elem in classes:
        if elem in test_classes:  # DONT ADD IF THE ACTION IS PART OF SUPPORT SET
            continue
        class_id = int(inv_class_dict[elem][1:])
        if 50 <= class_id <= 60 or class_id >= 106:  # DONT ADD IF ACTION INVOLVES TWO PERSON
            continue
        filtered_classes.append(elem)
    random.shuffle(filtered_classes)
    n_train = int(len(filtered_classes) * 0.8)
    train_data.classes = filtered_classes[:n_train]
    valid_data.classes = filtered_classes[n_train:]
    test_data.classes = test_classes

    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.n_workers)

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    scheduler = MultiStepLR(optimizer, milestones=[args.first_mile, args.second_mile],
                            gamma=0.1)

    # Start WANDB
    if ubuntu:
        run = wandb.init(project="trx", settings=wandb.Settings(start_method='fork'), config=args.__dict__)
        wandb.watch(model, log='all', log_freq=args.log_every)

    # Log
    print("Train samples: {}, valid samples: {}".format(len(train_loader), len(valid_loader)))
    print("Training for {} epochs".format(args.n_epochs))

    # Open set loss
    os_loss_fn = torch.nn.BCELoss()
    fs_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):

        fs_train_losses = []
        fs_train_accuracies = []
        os_train_losses = []
        os_train_pred = []
        os_train_true = []

        fs_valid_losses = []
        fs_valid_accuracies = []
        os_valid_losses = []
        os_valid_pred = []
        os_valid_true = []

        # TRAIN
        model.train()
        torch.set_grad_enabled(True)
        for i, elem in enumerate(tqdm(train_loader)):
            support_set, target_set, _, support_labels, target_label = elem
            batch_size = support_set.size(0)

            support_set = support_set.reshape(batch_size, args.way, args.seq_len, args.n_joints * 3).cuda().float()
            target_set = target_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            # unknown_set = unknown_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = support_labels.reshape(batch_size, args.way).cuda().int()
            target_label = target_label.cuda()

            ################
            # Known action #
            ################
            out = model(support_set, support_labels, target_set)

            # FS known
            fs_pred = out['logits']
            target = torch.zeros_like(fs_pred)
            target[torch.arange(batch_size), target_label.long()] = 1
            scores = fs_pred
            known_fs_loss = fs_loss_fn(fs_pred, target)
            fs_train_losses.append(known_fs_loss.item())
            final_loss = known_fs_loss

            train_accuracy = torch.eq(torch.argmax(scores, dim=1), target_label).int().float().mean().item()
            fs_train_accuracies.append(train_accuracy)

            if epoch > args.start_discriminator_after_epoch:
                # OS known (target depends on true class)
                os_pred = out['is_true']
                target = torch.eq(torch.argmax(fs_pred, dim=1), target_label).float().unsqueeze(-1)
                known_os_loss = os_loss_fn(os_pred, target)
                os_train_true.append(target.cpu().numpy())
                os_train_pred.append((os_pred > 0.5).float().cpu().numpy())
                os_train_losses.append(known_os_loss.item())

                # ##################
                # # Unknown action #
                # ##################
                # out = model(support_set, support_labels, unknown_set)
                #
                # # OS unknown
                # os_pred = out['is_true']
                # target = torch.zeros_like(os_pred)
                # unknown_os_loss = os_loss_fn(os_pred.squeeze(), target)
                #
                # os_train_losses.append(unknown_os_loss.item())
                # os_train_true.append(target.cpu().numpy())
                # os_train_pred.append((os_pred > 0.5).float())

                final_loss = final_loss + known_os_loss  # + unknown_os_loss
            ############
            # Optimize #
            ############
            final_loss.backward(retain_graph=True)

            if i % args.optimize_every == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        # EVAL
        model.eval()
        torch.set_grad_enabled(False)
        for i, elem in enumerate(tqdm(valid_loader)):
            support_set, target_set, _, support_labels, target_label = elem
            batch_size = support_set.size(0)

            support_set = support_set.reshape(batch_size, args.way, args.seq_len, args.n_joints * 3).cuda().float()
            target_set = target_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            # unknown_set = unknown_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = support_labels.reshape(batch_size, args.way).cuda().int()
            target_label = target_label.cuda()

            ################
            # Known action #
            ################
            out = model(support_set, support_labels, target_set)

            # FS known
            fs_pred = out['logits']
            target = torch.zeros_like(fs_pred)
            target[torch.arange(batch_size), target_label.long()] = 1
            scores = fs_pred
            known_fs_loss = fs_loss_fn(scores, target)
            fs_valid_losses.append(known_fs_loss.item())
            final_loss = known_fs_loss

            valid_accuracy = torch.eq(torch.argmax(scores, dim=1), target_label).int().float().mean().item()
            fs_valid_accuracies.append(valid_accuracy)

            if epoch > args.start_discriminator_after_epoch:
                # OS known (target depends on true class)
                os_pred = out['is_true']
                target = torch.eq(torch.argmax(fs_pred, dim=1), target_label).float().unsqueeze(-1)
                known_os_loss = os_loss_fn(os_pred, target)
                os_valid_true.append(target.cpu().numpy())
                os_valid_pred.append((os_pred > 0.5).float().cpu().numpy())
                os_valid_losses.append(known_os_loss.item())

                # ##################
                # # Unknown action #
                # ##################
                # out = model(support_set, support_labels, unknown_set)
                #
                # # OS unknown
                # os_pred = out['is_true']
                # target = torch.zeros_like(os_pred)
                # unknown_os_loss = os_loss_fn(os_pred.squeeze(), target)
                #
                # os_valid_losses.append(unknown_os_loss.item())
                # os_valid_true.append(target.cpu().numpy())
                # os_valid_pred.append((os_pred > 0.5).float())

        # WANDB
        os_train_true = np.concatenate(os_train_true, axis=None) if len(os_train_true) > 0 else np.zeros(1)
        os_train_pred = np.concatenate(os_train_pred, axis=None) if len(os_train_pred) > 0 else np.zeros(1)
        os_valid_true = np.concatenate(os_valid_true, axis=None) if len(os_valid_true) > 0 else np.zeros(1)
        os_valid_pred = np.concatenate(os_valid_pred, axis=None) if len(os_valid_pred) > 0 else np.zeros(1)
        epoch_path = checkpoints_path + os.sep + '{}.pth'.format(epoch)
        if ubuntu:
            wandb.log({"train/fs_loss": sum(fs_train_losses) / len(fs_train_losses),
                       "train/fs_accuracy": sum(fs_train_accuracies) / len(fs_train_accuracies),
                       "train/os_loss": (sum(os_train_losses) / len(os_train_losses)) if len(os_train_losses) > 0 else 0,
                       "train/os_accuracy": accuracy_score(os_train_true, os_train_pred),
                       "train/os_precision": precision_score(os_train_true, os_train_pred, zero_division=0),
                       "train/os_recall": recall_score(os_train_true, os_train_pred, zero_division=0),
                       "train/os_f1": f1_score(os_train_true, os_train_pred, zero_division=0),
                       "train/os_n_1_true": os_train_true.mean(),
                       "train/os_n_1_pred": os_train_pred.mean(),

                       "valid/fs_loss": sum(fs_valid_losses) / len(fs_valid_losses),
                       "valid/fs_accuracy": sum(fs_valid_accuracies) / len(fs_valid_accuracies),
                       "valid/os_loss": (sum(os_valid_losses) / len(os_valid_losses)) if len(os_valid_losses) > 0 else 0,
                       "valid/os_accuracy": accuracy_score(os_valid_true, os_valid_pred),
                       "valid/os_precision": precision_score(os_valid_true, os_valid_pred, zero_division=0),
                       "valid/os_recall": recall_score(os_valid_true, os_valid_pred, zero_division=0),
                       "valid/os_f1": f1_score(os_valid_true, os_valid_pred, zero_division=0),
                       "valid/os_n_1_true": os_valid_true.mean(),
                       "valid/os_n_1_pred": os_valid_pred.mean(),

                       "lr": optimizer.param_groups[0]['lr']})

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fs_accuracy': sum(fs_valid_accuracies) / len(fs_valid_accuracies),
                    'os_accuracy': f1_score(os_valid_true, os_valid_pred, zero_division=0)},
                   epoch_path)

    if ubuntu:
        run.join()
