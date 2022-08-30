import cv2
from torch import nn
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
import torch

# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=8G bash

gpu_id = 1 if ubuntu else 0
torch.cuda.set_device(gpu_id)
torch.manual_seed(0)
device = TRXConfig().device

if __name__ == "__main__":
    args = TRXConfig()

    # Get conversion class id -> class label
    test_classes = ["A1", "A7", "A13", "A19", "A25", "A31", "A37", "A43", "A49",
                    # "A55", # two person
                    "A61", "A67", "A73", "A79", "A85", "A91", "A97", "A103",
                    # "A109", "A115" # two person
                    ]
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
    model.load_state_dict(torch.load('modules/ar/modules/raws/rgb/5000.pth')['model_state_dict'])
    # TODO MAKE IT WORK IN EVAL MODE
    # model.train()
    model.eval()
    # count = 0
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         count += 1  # skip the first BatchNorm layer in my ResNet50 based encoder
    #         if count >= 2:
    #             m.eval()
    #             m.weight.requires_grad = False
    #             m.bias.requires_grad = False
    # TODO MAKE IT WORK IN EVAL MODE

    # Create dataset iterator
    train_data = EpisodicLoader(args.data_path, k=args.way, n_task=100, input_type=args.input_type, )

    # Divide dataset into train and validation
    train_data.classes = test_classes
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers,
                                               shuffle=True)

    # Log
    print("Train samples: {}".format(len(train_loader)))
    print("Training for {} epochs".format(args.n_epochs))
    print("Batch size is {}".format(args.batch_size))

    # Losses
    os_loss_fn = torch.nn.BCELoss()
    fs_loss_fn = torch.nn.CrossEntropyLoss()

    fs_train_losses = []
    fs_train_accuracies = []
    os_train_losses = []
    os_train_pred = []
    os_train_true = []

    with torch.no_grad():
        for elem in tqdm(train_loader):

            support_set = elem['support_set'].float().to(device)
            target_set = elem['target_set'].float().to(device)
            unknown_set = elem['unknown_set'].float().to(device)

            batch_size = support_set.size(0)

            # TODO START MOVE IN DATALOADER
            # support_set = support_set.reshape(batch_size, args.way, args.seq_len, args.n_joints * 3).cuda().float()
            # target_set = target_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            # unknown_set = unknown_set.reshape(batch_size, args.seq_len, args.n_joints * 3).cuda().float()
            support_set = torch.permute(support_set, (0, 1, 2, 5, 3, 4))
            target_set = torch.permute(target_set, (0, 1, 4, 2, 3))
            unknown_set = torch.permute(unknown_set, (0, 1, 4, 2, 3))
            # TODO END MOVE IN DATALOADER

            support_labels = torch.arange(args.way).repeat(batch_size).reshape(batch_size, args.way).to(device).int()
            target = np.array(elem['support_classes']).T == \
                     np.repeat(np.array(elem['target_class']), 5).reshape(batch_size, args.way)
            target = torch.FloatTensor(target).to(device)

            ################
            # Known action #
            ################
            out = model(support_set, support_labels, target_set)

            # FS known
            fs_pred = out['logits']
            known_fs_loss = fs_loss_fn(fs_pred, target)
            fs_train_losses.append(known_fs_loss.item())

            train_accuracy = torch.eq(torch.argmax(fs_pred, dim=1), torch.argmax(target, dim=1)).float().mean().item()
            fs_train_accuracies.append(train_accuracy)

            # # TODO START VISUAL DEBUG
            # print(elem['support_classes'])
            # print(elem['target_class'])
            # print(out['logits'].detach().cpu().numpy())
            #
            # support_set = (support_set.permute(0, 1, 2, 4, 5, 3) - torch.FloatTensor([0.485, 0.456, 0.406]).cuda()) / torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
            # support_set = support_set.detach().cpu().numpy()
            # support_set = (support_set * 255).astype(int)
            # support_set = support_set[0].swapaxes(0, 1).reshape(8, 224*5, 224, 3).swapaxes(0, 1).reshape(5*224, 8*224, 3)
            #
            # target_set = (target_set.permute(0, 1, 3, 4, 2) - torch.FloatTensor([0.485, 0.456, 0.406]).cuda()) / torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
            # target_set = target_set.detach().cpu().numpy()
            # target_set = (target_set * 255).astype(int)
            # target_set = target_set[0].swapaxes(0, 1).reshape(224, -1, 3)
            #
            # cv2.imwrite("support.png", support_set)
            # cv2.imwrite("target.png", target_set)
            # with open("result.txt", "w") as outfile:
            #     outfile.write("%s \n %s \n %s \n" % (elem['support_classes'], elem['target_class'], out['logits'].detach().cpu().numpy()))
            # # cv2.imshow("support_set", support_set)
            # # cv2.imshow("target_set", target_set)
            # # cv2.waitKey(0)
            # # TODO END VISUAL DEBUG
            #
            # input()

            # print("Loss is", known_fs_loss.item(), "out is", fs_pred.detach().cpu().numpy(), "target:", target.detach().cpu().numpy())

            # if epoch > args.start_discriminator_after_epoch:
            #     # OS known (target depends on true class)
            #     os_pred = out['is_true']
            #     target = torch.eq(torch.argmax(fs_pred, dim=1), torch.argmax(target, dim=1)).float().unsqueeze(-1)
            #     # Train only on correct prediction
            #     # true_s = (target == 1.).nonzero(as_tuple=True)[0]
            #     # n = len(true_s)
            #     # os_pred = os_pred[true_s]
            #     # target = target[true_s]
            #     if target.item() == 1:  # TODO do something like this to train discriminator with batch size = 1
            #         known_os_loss = os_loss_fn(os_pred, target)
            #         os_train_true.append(target.cpu().numpy())
            #         os_train_pred.append((os_pred > 0.5).float().cpu().numpy())
            #         os_train_losses.append(known_os_loss.item())
            #
            #         # TODO EXPERIMENT TO TRAIN RGB
            #         known_os_loss.backward()
            #         if i % args.optimize_every == 0:
            #             optimizer.step()
            #             optimizer.zero_grad()
            #         # TODO END EXPERIMENT TO TRAIN RGB
            #
            #         ##################
            #         # Unknown action #
            #         ##################
            #         out = model(support_set, support_labels, unknown_set)
            #
            #         # OS unknown
            #         os_pred = out['is_true']
            #         target = torch.zeros_like(os_pred)
            #         # Get n samples
            #         # os_pred = os_pred[:n]
            #         # target = target[:n]
            #
            #         unknown_os_loss = os_loss_fn(os_pred, target)
            #         os_train_losses.append(unknown_os_loss.item())
            #         os_train_true.append(target.cpu().numpy())
            #         os_train_pred.append((os_pred > 0.5).float().cpu().numpy())
            #
            #         # TODO EXPERIMENT TO TRAIN RGB
            #         unknown_os_loss.backward()
            #         if i % args.optimize_every == 0:
            #             optimizer.step()
            #             optimizer.zero_grad()
            #         # TODO END EXPERIMENT TO TRAIN RGB
            #
            #     final_loss = final_loss + known_os_loss + unknown_os_loss
            ############
            # Optimize #
            ############

            # WANDB

            # known_fs_loss.backward()  # To free memory

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
           "train/os_n_1_pred": os_train_pred.mean()})
