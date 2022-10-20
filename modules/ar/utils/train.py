import os
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
from modules.ar.utils.dataloader import EpisodicLoader, MyLoader
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig
from utils.params import ubuntu
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np


# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash
# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gpus-per-node=4 --cpus-per-task=32 --mem=8G bash

# gpu_id = 1 if ubuntu else 0
# torch.cuda.set_device(gpu_id)
# torch.manual_seed(0)
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

    # Create checkpoints directory
    if not os.path.exists(args.checkpoints_path):
        os.mkdir(args.checkpoints_path)
    checkpoints_path = args.checkpoints_path + os.sep + datetime.now().strftime("%d_%m_%Y-%H_%M")
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    # Get model
    model = TRXOS(args).to(device)
    if args.input_type in ["rgb", "hybrid"] and ubuntu:
        print("Model has been distributed over multiple GPUs")
        model.distribute_model()

    # Create dataset iterator
    train_data = MyLoader(args.data_path, k=args.way, n_task=args.n_task, input_type=args.input_type, l=args.seq_len)
    valid_data = MyLoader(args.data_path, k=args.way, n_task=args.n_task, input_type=args.input_type, l=args.seq_len)

    all_classes = list(filter(lambda x: x not in test_classes, train_data.all_classes))
    idx = int(len(all_classes)*0.8)
    train_data.all_classes, valid_data.all_classes = all_classes[:idx], all_classes[idx:]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.n_workers,
                                               shuffle=True)

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_lr)
    optimizer.zero_grad()
    scheduler = MultiStepLR(optimizer, milestones=[args.first_mile, args.second_mile],
                            gamma=0.1)

    # Start WANDB
    if ubuntu:
        run = wandb.init(project="trx", settings=wandb.Settings(start_method='fork'), config=args.__dict__)
        wandb.watch(model, log='all', log_freq=args.log_every)

    # Log
    print("Input type:", args.input_type)
    print("Train samples: {}".format(len(train_loader)))
    print("Valid samples: {}".format(len(valid_loader)))
    print("Training for {} epochs".format(args.n_epochs))
    print("Batch size is {}".format(args.batch_size))
    print(f"Using {args.n_workers} workers")
    print(f"Eval is done every {args.eval_every_n_epoch} epochs")

    # Losses
    os_loss_fn = torch.nn.BCELoss()
    fs_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):

        fs_losses = []
        fs_accuracies = []
        os_losses = []
        os_preds = []
        os_trues = []
        os_outs = []
        do_eval = (epoch % 10) == 0

        # TRAIN
        model.train() if not do_eval else model.eval()
        for i, elem in enumerate(tqdm(train_loader if not do_eval else valid_loader)):

            # Extract from dict, convert, move to GPU
            support_set = {t: elem['support_set'][t].float().to(device) for t in elem['support_set'].keys()}
            target_set = {t: elem['target_set'][t].float().to(device) for t in elem['target_set'].keys()}
            unknown_set = {t: elem['unknown_set'][t].float().to(device) for t in elem['unknown_set'].keys()}

            support_labels = torch.arange(args.way).repeat(b).reshape(b, args.way).to(device).int()
            target = (elem['support_classes'] == elem['target_class'][..., None]).float().to(device)

            ################
            # Known action #
            ################
            out = model(support_set, support_labels, target_set)

            # FS known
            fs_pred = out['logits']
            known_fs_loss = fs_loss_fn(fs_pred, target)
            fs_losses.append(known_fs_loss.item())
            final_loss = known_fs_loss

            train_accuracy = torch.eq(torch.argmax(fs_pred, dim=1), torch.argmax(target, dim=1)).float().mean().item()
            fs_accuracies.append(train_accuracy)

            if epoch > args.start_discriminator_after_epoch-1:
                # OS known (target depends on true class)
                os_pred = out['is_true']
                os_outs.append(os_pred.detach().cpu().numpy())
                target = torch.eq(torch.argmax(fs_pred, dim=1), torch.argmax(target, dim=1)).float().unsqueeze(-1)
                # Train only on correct prediction
                true_s = (target == 1.).nonzero(as_tuple=True)[0]
                n = len(true_s)
                os_pred = os_pred[true_s]
                target = target[true_s]

                if target.sum() > 0:  # TODO do something like this to train discriminator with batch size = 1
                    known_os_loss = os_loss_fn(os_pred, target)
                    os_trues.append(target.cpu().numpy())
                    os_preds.append((os_pred > 0.5).float().cpu().numpy())
                    os_losses.append(known_os_loss.item())

                    # TODO EXPERIMENT TO TRAIN RGB
                    # known_os_loss.backward()
                    # if i % args.optimize_every == 0:
                    #     optimizer.step()
                    #     optimizer.zero_grad()
                    # TODO END EXPERIMENT TO TRAIN RGB

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
                    os_losses.append(unknown_os_loss.item())
                    os_trues.append(target.cpu().numpy())
                    os_preds.append((os_pred > 0.5).float().cpu().numpy())

                    # TODO EXPERIMENT TO TRAIN RGB
                    # unknown_os_loss.backward()
                    # if i % args.optimize_every == 0:
                    #     optimizer.step()
                    #     optimizer.zero_grad()
                    # TODO END EXPERIMENT TO TRAIN RGB

                    final_loss = final_loss + known_os_loss + unknown_os_loss
            ############
            # Optimize #
            ############
            final_loss = final_loss / args.optimize_every
            if not do_eval:
                final_loss.backward()

                if i % args.optimize_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        if not do_eval:
            scheduler.step()

        # WANDB
        os_trues = np.concatenate(os_trues, axis=None) if len(os_trues) > 0 else np.zeros(1)
        os_preds = np.concatenate(os_preds, axis=None) if len(os_preds) > 0 else np.zeros(1)
        epoch_path = checkpoints_path + os.sep + '{}.pth'.format(epoch)
        if ubuntu:
            lbl = "train" if not do_eval else "valid"
            wandb.log({lbl+"/fs_loss": sum(fs_losses) / len(fs_losses),
                       lbl+"/fs_accuracy": sum(fs_accuracies) / len(fs_accuracies),
                       lbl+"/os_loss": (sum(os_losses) / len(os_losses)) if len(os_losses) > 0 else 0,
                       lbl+"/os_accuracy": accuracy_score(os_trues, os_preds),
                       lbl+"/os_precision": precision_score(os_trues, os_preds, zero_division=0),
                       lbl+"/os_recall": recall_score(os_trues, os_preds, zero_division=0),
                       lbl+"/os_f1": f1_score(os_trues, os_preds, zero_division=0),
                       lbl+"/os_n_1_true": os_trues.mean(),
                       lbl+"/os_n_1_pred": os_preds.mean(),
                       "lr": optimizer.param_groups[0]['lr'],
                       "os_outs": wandb.Histogram(np.concatenate(os_outs, axis=0)) if len(os_outs) > 0 else [0]})

        if not do_eval:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       epoch_path)

    if ubuntu:
        run.join()
