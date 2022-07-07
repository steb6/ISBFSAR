import os
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
from modules.ar.utils.dataloader import EpisodicLoader
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig
from utils.params import ubuntu
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np


# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash

device = 1 if ubuntu else 0
torch.cuda.set_device(device)
torch.manual_seed(0)


if __name__ == "__main__":
    args = TRXConfig()

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

    # Create checkpoints directory
    if not os.path.exists(args.checkpoints_path):
        os.mkdir(args.checkpoints_path)
    checkpoints_path = args.checkpoints_path + os.sep + datetime.now().strftime("%d_%m_%Y-%H_%M")
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    # Get model
    model = TRXOS(args).cuda(device)
    model.train()

    # Create dataset iterator
    train_data = EpisodicLoader(args.data_path, k=args.way, n_task=10000 if ubuntu else 100, input_type=args.input_type)

    # Divide dataset into train and validation
    classes = train_data.classes
    inv_class_dict = {v: k for k, v in class_dict.items()}
    filtered_classes = []
    for elem in classes:
        if elem in test_classes:  # DONT ADD IF THE ACTION IS PART OF SUPPORT SET
            continue
        class_id = int(inv_class_dict[elem][1:])
        if "metrabs" in args.data_path:  # we dont have two actions for metrabs, but for NTURGBD yes
            if 50 <= class_id <= 60 or class_id >= 106:  # DONT ADD IF ACTION INVOLVES TWO PERSON
                continue
        filtered_classes.append(elem)
    train_data.classes = filtered_classes
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers,
                                               shuffle=True)

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
    print("Train samples: {}".format(len(train_loader)))
    print("Training for {} epochs".format(args.n_epochs))

    # Losses
    os_loss_fn = torch.nn.BCELoss()
    fs_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):

        fs_train_losses = []
        fs_train_accuracies = []
        os_train_losses = []
        os_train_pred = []
        os_train_true = []

        # TRAIN
        model.train()
        torch.set_grad_enabled(True)
        for i, elem in enumerate(tqdm(train_loader)):

            support_set = elem['support_set']
            target_set = elem['target_set']
            unknown_set = elem['unknown_set']

            batch_size = support_set.size(0)

            support_set = support_set.reshape(batch_size, args.way, args.seq_len, args.n_joints * 3).cuda().float()
            target_set = target_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            unknown_set = unknown_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = torch.arange(args.way).repeat(batch_size).reshape(batch_size, args.way).cuda().int()
            target = np.array(elem['support_classes']).T == \
                     np.repeat(np.array(elem['target_class']), 5).reshape(batch_size, args.way)
            target = torch.FloatTensor(target).cuda()

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

            if epoch > args.start_discriminator_after_epoch:
                # OS known (target depends on true class)
                os_pred = out['is_true']
                target = torch.eq(torch.argmax(fs_pred, dim=1), torch.argmax(target, dim=1)).float().unsqueeze(-1)
                # Train only on correct prediction
                true_s = (target == 1.).nonzero(as_tuple=True)[0]
                n = len(true_s)
                os_pred = os_pred[true_s]
                target = target[true_s]

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
                target = torch.zeros_like(os_pred)
                # Get n samples
                os_pred = os_pred[:n]
                target = target[:n]

                unknown_os_loss = os_loss_fn(os_pred, target)
                os_train_losses.append(unknown_os_loss.item())
                os_train_true.append(target.cpu().numpy())
                os_train_pred.append((os_pred > 0.5).float().cpu().numpy())

                final_loss = final_loss + known_os_loss + unknown_os_loss
            ############
            # Optimize #
            ############
            final_loss.backward(retain_graph=True)

            if i % args.optimize_every == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # WANDB
        os_train_true = np.concatenate(os_train_true, axis=None) if len(os_train_true) > 0 else np.zeros(1)
        os_train_pred = np.concatenate(os_train_pred, axis=None) if len(os_train_pred) > 0 else np.zeros(1)
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
                       "lr": optimizer.param_groups[0]['lr']})

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   epoch_path)

    if ubuntu:
        run.join()
