import os
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
from modules.ar.trx.utils.dataloader import MetrabsData
from modules.ar.trx.utils.misc import aggregate_accuracy, loss_fn
from modules.ar.trx.utils.model import CNN_TRX
from utils.params import TRXConfig


# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash


log_every = 1000
optimize_every = 16

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

    data_path = "D:\\nturgbd_metrabs" if 'Users' in os.getcwd() else "nturgbd_metrabs_2"  # TODO ADJUST
    train_data = MetrabsData(data_path, k=5, mode='train', n_task=10000)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=12)
    valid_data = MetrabsData(data_path, k=5, mode='valid', n_task=1000)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, num_workers=12)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    scheduler = MultiStepLR(optimizer, milestones=[10000, 100000], gamma=0.1)

    run = wandb.init(project="trx")
    wandb.watch(model, log='all', log_freq=log_every)

    for epoch in range(10000):

        # TRAIN
        model.train()
        train_losses = []
        train_accuracies = []
        for i, elem in enumerate(tqdm(train_loader)):
            torch.set_grad_enabled(True)

            support_set, target_set, support_labels, target_label = elem

            support_set = support_set.reshape(args.way * args.seq_len, args.n_joints * 3).cuda().float()
            target_set = target_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = support_labels.reshape(args.way).cuda().int()
            target_label = target_label.cuda()

            out = model(support_set, support_labels, target_set)
            pred = out['logits']

            target = torch.zeros_like(support_labels)
            target[target_label.item()] = 1.

            train_loss = loss_fn(pred.unsqueeze(0), target.unsqueeze(0), 'cuda')
            # loss = loss / 16
            train_loss.backward(retain_graph=False)
            train_losses.append(train_loss.item())

            train_accuracy = aggregate_accuracy(pred, target_label)
            train_accuracies.append(train_accuracy.item())

            if i % optimize_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if i % log_every == 0:
                wandb.log({"train/loss": sum(train_losses) / len(train_losses),
                           "train/accuracy": sum(train_accuracies) / len(train_accuracies),
                           "lr": optimizer.param_groups[0]['lr']})
                train_losses = []
                train_accuracies = []

        # EVAL
        model.eval()
        torch.set_grad_enabled(False)
        valid_losses = []
        valid_accuracies = []
        for i, elem in enumerate(tqdm(valid_loader)):
            support_set, target_set, support_labels, target_label = elem

            support_set = support_set.reshape(args.way * args.seq_len, args.n_joints * 3).cuda().float()
            target_set = target_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
            support_labels = support_labels.reshape(args.way).cuda().int()
            target_label = target_label.cuda()

            out = model(support_set, support_labels, target_set)
            pred = out['logits']

            target = torch.zeros_like(support_labels)
            target[target_label.item()] = 1.

            valid_loss = loss_fn(pred.unsqueeze(0), target.unsqueeze(0), 'cuda')
            valid_losses.append(valid_loss.item())

            valid_accuracy = aggregate_accuracy(pred, target_label)
            valid_accuracies.append(valid_accuracy.item())

        # WANDB
        epoch_path = checkpoints_path + os.sep + '{}.pth'.format(epoch)
        wandb.log({"valid/loss": sum(valid_losses) / len(valid_losses) if len(valid_losses) > 0 else -1,
                   "valid/accuracy": sum(valid_accuracies) / len(valid_accuracies) if len(valid_losses) > 0 else -1,
                   "lr": optimizer.param_groups[0]['lr']})

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': sum(valid_losses) / len(valid_losses) if len(valid_losses) > 0 else -1},
                   epoch_path)

        # artifact.add_file(epoch_path)
        # run.log_artifact(artifact)

    run.join()
