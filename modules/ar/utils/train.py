import os
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
from modules.ar.utils.dataloader import MetrabsData
from modules.ar.utils.misc import aggregate_accuracy, loss_fn
from modules.ar.utils.model import CNN_TRX
from utils.params import TRXConfig
import random
from utils.params import ubuntu


# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash

if __name__ == "__main__":
    args = TRXConfig()
    torch.manual_seed(0)

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    checkpoints_path = "checkpoints" + os.sep + datetime.now().strftime("%d_%m_%Y-%H_%M")
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    device = 'cuda:0'
    model = CNN_TRX(args).to(device)
    model.train()

    # Create dataset iterator
    train_data = MetrabsData(args.data_path, k=5, n_task=10000)
    valid_data = MetrabsData(args.data_path, k=5, n_task=1000)

    # Divide dataset into train and validation
    classes = train_data.classes
    random.shuffle(classes)
    n_train = int(len(classes) * 0.8)
    train_data.classes = classes[:n_train]
    valid_data.classes = classes[n_train:]

    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=args.n_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, num_workers=args.n_workers)

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    scheduler = MultiStepLR(optimizer, milestones=[10000, 100000], gamma=0.1)

    # Start WANDB
    if ubuntu:
        run = wandb.init(project="trx")
        wandb.watch(model, log='all', log_freq=args.log_every)

    # Log
    print("Train samples: {}, valid samples: {}".format(len(train_loader), len(valid_loader)))
    print("Training for {} epochs".format(args.n_epochs))

    # Open set loss
    os_loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.n_epochs):

        fs_train_losses = []
        fs_train_accuracies = []
        os_train_losses = []
        os_train_accuracies = []

        fs_valid_losses = []
        fs_valid_accuracies = []
        os_valid_losses = []
        os_valid_accuracies = []

        # TRAIN
        model.train()
        torch.set_grad_enabled(True)
        for i, elem in enumerate(tqdm(train_loader)):
            support_set, target_set, unknown_set, support_labels, target_label = elem

            support_set = support_set.reshape(args.way * args.seq_len, args.n_joints * 3).cuda().float()
            target_set = target_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
            unknown_set = unknown_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = support_labels.reshape(args.way).cuda().int()
            target_label = target_label.cuda()

            ################
            # Known action #
            ################
            out = model(support_set, support_labels, target_set)

            # FS known
            pred = out['logits']

            target = torch.zeros_like(support_labels)
            target[target_label.item()] = 1.

            known_fs_loss = loss_fn(pred.unsqueeze(0), target.unsqueeze(0), 'cuda')
            fs_train_losses.append(known_fs_loss.item())

            train_accuracy = aggregate_accuracy(pred, target_label)
            fs_train_accuracies.append(train_accuracy.item())

            # OS known
            pred = out['is_true']
            known_os_loss = os_loss_fn(pred, torch.FloatTensor([1.]).unsqueeze(0).cuda())
            os_train_losses.append(known_os_loss.item())
            os_train_accuracies.append(1. if pred.item() > 0.5 else 0)

            ##################
            # Unknown action #
            ##################
            out = model(support_set, support_labels, unknown_set)

            # OS unknown
            pred = out['is_true']
            unknown_os_loss = os_loss_fn(pred, torch.FloatTensor([0.]).unsqueeze(0).cuda())
            os_train_losses.append(unknown_os_loss.item())
            os_train_accuracies.append(1. if pred.item() < 0.5 else 0)

            ############
            # Optimize #
            ############
            final_loss = known_fs_loss + known_os_loss + unknown_os_loss
            final_loss.backward(retain_graph=True)

            if i % args.optimize_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # EVAL
        model.eval()
        torch.set_grad_enabled(False)
        for i, elem in enumerate(tqdm(valid_loader)):
            support_set, target_set, unknown_set, support_labels, target_label = elem

            support_set = support_set.reshape(args.way * args.seq_len, args.n_joints * 3).cuda().float()
            target_set = target_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
            unknown_set = unknown_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = support_labels.reshape(args.way).cuda().int()
            target_label = target_label.cuda()

            ################
            # Known action #
            ################
            out = model(support_set, support_labels, target_set)

            # FS known
            pred = out['logits']

            target = torch.zeros_like(support_labels)
            target[target_label.item()] = 1.

            known_fs_loss = loss_fn(pred.unsqueeze(0), target.unsqueeze(0), 'cuda')
            fs_valid_losses.append(known_fs_loss.item())

            valid_accuracy = aggregate_accuracy(pred, target_label)
            fs_valid_accuracies.append(valid_accuracy.item())

            # OS known
            pred = out['is_true']
            known_os_loss = os_loss_fn(pred, torch.FloatTensor([1.]).unsqueeze(0).cuda())
            os_valid_losses.append(known_os_loss.item())
            os_valid_accuracies.append(1. if pred.item() > 0.5 else 0)

            ##################
            # Unknown action #
            ##################
            out = model(support_set, support_labels, unknown_set)

            # OS unknown
            pred = out['is_true']
            unknown_os_loss = os_loss_fn(pred, torch.FloatTensor([0.]).unsqueeze(0).cuda())
            os_valid_losses.append(unknown_os_loss.item())
            os_valid_accuracies.append(1. if pred.item() < 0.5 else 0)

        # WANDB
        epoch_path = checkpoints_path + os.sep + '{}.pth'.format(epoch)
        if ubuntu:
            wandb.log({"train/fs_loss": sum(fs_train_losses) / len(fs_train_losses),
                       "train/fs_accuracy": sum(fs_train_accuracies) / len(fs_train_accuracies),
                       "train/os_loss": sum(os_train_losses) / len(os_train_losses),
                       "train/os_accuracy": sum(os_train_accuracies) / len(os_train_accuracies),
                       "valid/fs_loss": sum(fs_valid_losses) / len(fs_valid_losses),
                       "valid/fs_accuracy": sum(fs_valid_accuracies) / len(fs_valid_accuracies),
                       "valid/os_loss": sum(os_valid_losses) / len(os_valid_losses),
                       "valid/os_accuracy": sum(os_valid_accuracies) / len(os_valid_accuracies),
                       "lr": optimizer.param_groups[0]['lr']})

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fs_accuracy': sum(fs_valid_accuracies) / len(fs_valid_accuracies),
                    'os_accuracy': sum(os_valid_accuracies) / len(os_valid_accuracies)},
                   epoch_path)

    if ubuntu:
        run.join()
