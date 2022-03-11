import os
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
from utils.dataloader import MetrabsData
from utils.misc import aggregate_accuracy, loss_fn
from utils.model import CNN_TRX

# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash

if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 1
            self.trans_dropout = 0.
            self.seq_len = 16
            self.n_joints = 30
            self.num_gpus = 1
            self.temp_set = [2, 3]


    args = ArgsObject()
    torch.manual_seed(0)

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    checkpoints_path = "checkpoints" + os.sep + datetime.now().strftime("%d_%m_%Y-%H_%M")
    os.mkdir(checkpoints_path)

    device = 'cuda:0'
    model = CNN_TRX(args).to(device)
    model.train()

    # EVAL
    model.eval()
    torch.set_grad_enabled(False)
    valid_losses = []
    valid_accuracies = []
    while True:
        support_set, target_set, support_labels, target_label = elem

        support_set = support_set.reshape(args.way * args.seq_len, args.n_joints * 3).cuda().float()
        target_set = target_set.reshape(args.seq_len, args.n_joints * 3).cuda().float()
        support_labels = support_labels.reshape(args.way).cuda().float()
        target_label = target_label.cuda()

        out = model(support_set, support_labels, target_set)
        pred = out['logits']

        target = torch.zeros_like(support_labels)
        target[target_label.item()] = 1.

        valid_loss = loss_fn(pred, target.unsqueeze(0), 'cuda')
        valid_losses.append(valid_loss.item())

        valid_accuracy = aggregate_accuracy(pred, target_label)
        valid_accuracies.append(valid_accuracy.item())
