from datetime import datetime
import torch.optim
from torch.nn import BCELoss
import wandb
from modules.focus.mutual_gaze.focus_detection.utils.my_dataloader import MARIAData
from modules.focus.mutual_gaze.focus_detection.utils.model import JustOpenPose as Model
from tqdm import tqdm
from sklearn import metrics
import platform
import numpy as np
from utils.params import MutualGazeConfig
import os

if __name__ == "__main__":

    is_local = "Windows" in platform.platform()

    config = MutualGazeConfig()

    if not os.path.exists(config.ckpts_path):
        os.mkdir(config.ckpts_path)
    ckpts_path = os.path.join(config.ckpts_path, datetime.now().strftime("%d_%m_%Y-%H_%M"))
    os.mkdir(ckpts_path)

    all_losses = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for sess in range(5):
        train_data = MARIAData(config.data_path, mode="train", split_number=sess)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                                   num_workers=2 if is_local else 12, shuffle=True)
        valid_data = MARIAData(config.data_path, mode="valid", split_number=sess)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32,
                                                   num_workers=2 if is_local else 4)
        test_data = MARIAData(config.data_path, mode="test", split_number=sess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                                  num_workers=2 if is_local else 2)

        model = Model()
        model.cuda()
        model.train()

        loss_fn = BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-5)

        run = wandb.init(project="mutual_gaze", reinit=True, name="{}".format(sess),
                         settings=wandb.Settings(start_method='thread'), config=config.__dict__)
        wandb.watch(model, log='all', log_freq=config.log_every)

        print("Train set length: {}, valid set length: {}".format(len(train_loader) * config.batch_size,
                                                                  len(valid_loader) * config.batch_size))
        print("Train set watching: {}, not watching: {}".format(train_data.n_watch, train_data.n_not_watch))
        print("Valid set watching: {}, not watching: {}".format(valid_data.n_watch, valid_data.n_not_watch))
        print(train_data.sessions)
        print(valid_data.sessions)
        print(test_data.sessions)

        # for k in range(2):  # Try to unfreeze layers:
        #
        #     if k == 1:
        #         # model.load_state_dict(best_model)
        #         for parameter in list(model.backbone.parameters())[-2:]:
        #             parameter.requires_grad = True
        #         optimizer.param_groups[0]['params'] += list(model.backbone.parameters())[-2:]

        # patience = config.patience
        # Unfreeze a layer after 200 epoch
        # after = 100
        # how_many = 10
        # layers = 0
        best_f1 = 0
        last_valid_loss = 10000
        best_valid_loss = 10000
        best_model = None

        for epoch in range(config.n_epochs):
            # if epoch % after == 0 and epoch > 0:
            #     for k in range(how_many):
            #         list(model.backbone.parameters())[(-int(epoch/after)*how_many)-k].requires_grad = True
            #         layers += 1

            # TRAIN
            train_losses = []
            outs_train = []
            trues_train = []
            model.train()
            for img, pose, img_, x, y in tqdm(train_loader, desc="Train epoch {}".format(epoch)):
                x = x.cuda().float()
                y = y.cuda().float()

                out = model(x)
                outs_train.append(out.detach().cpu().numpy().reshape(-1))
                trues_train.append(y.detach().cpu().numpy().reshape(-1))
                loss = loss_fn(out, y.float().unsqueeze(-1))
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # EVAL
            with torch.no_grad():
                valid_losses = []
                model.eval()
                outs_valid = []
                trues_valid = []
                for img, pose, img_, x, y in tqdm(valid_loader, desc="Valid epoch {}".format(epoch)):
                    x = x.cuda().float()
                    y = y.cuda().float()

                    out = model(x)
                    outs_valid.append(out.detach().cpu().numpy())
                    trues_valid.append(y.detach().cpu().numpy())
                    loss = loss_fn(out, y.float().unsqueeze(-1))

                    valid_losses.append(loss.item())

            outs_train = np.concatenate(outs_train, axis=0).reshape(-1)
            trues_train = np.concatenate(trues_train, axis=0).reshape(-1)
            outs_valid = np.concatenate(outs_valid, axis=0).reshape(-1)
            trues_valid = np.concatenate(trues_valid, axis=0).reshape(-1)

            wandb.log({"train/loss": sum(train_losses) / len(train_losses),
                       "train/accuracy": metrics.accuracy_score(trues_train > 0.5, outs_train > 0.5),
                       "train/precision": metrics.precision_score(trues_train > 0.5, outs_train > 0.5, zero_division=0),
                       "train/recall": metrics.recall_score(trues_train > 0.5, outs_train > 0.5, zero_division=0),
                       "train/f1": metrics.f1_score(trues_train > 0.5, outs_train > 0.5, zero_division=0),
                       "valid/loss": sum(valid_losses) / len(valid_losses),
                       "valid/accuracy": metrics.accuracy_score(trues_valid > 0.5, outs_valid > 0.5),
                       "valid/precision": metrics.precision_score(trues_valid > 0.5, outs_valid > 0.5, zero_division=0),
                       "valid/recall": metrics.recall_score(trues_valid > 0.5, outs_valid > 0.5, zero_division=0),
                       "valid/f1": metrics.f1_score(trues_valid > 0.5, outs_valid > 0.5, zero_division=0),
                       "lr": optimizer.param_groups[0]['lr'],
                       "predicted": wandb.Histogram(outs_train),
                       "layers": 0})

            # Check if this is the best model
            score = metrics.f1_score(trues_valid > 0.5, outs_valid > 0.5)
            if score > best_f1:
                best_f1 = score
                best_model = model.state_dict()

            # Check patience
            # valid_loss = sum(valid_losses) / len(valid_losses)
            # if valid_loss > best_valid_loss:
            #     patience -= 1
            #     if patience == 0:
            #         break  # Exit the training loop
            # else:
            #     best_valid_loss = valid_loss
            #     patience = config.patience

        # TEST
        torch.save(best_model, os.path.join(ckpts_path, "sess_{}_f1_{:.2f}.pth".format(sess, best_f1)))
        test_losses = []
        outs_test = []
        trues_test = []
        model.load_state_dict(best_model)
        model.eval()
        for img, pose, img_, x, y in tqdm(valid_loader, desc="Test session {}".format(sess)):
            x = x.cuda().float()
            y = y.cuda().float()

            out = model(x)

            outs_test.append(out.detach().cpu().numpy().reshape(-1))
            trues_test.append(y.detach().cpu().numpy().reshape(-1))
            loss = loss_fn(out, y.float().unsqueeze(-1))
            loss.backward()
            optimizer.step()

            test_losses.append(loss.item())

        outs_test = np.concatenate(outs_test, axis=0).reshape(-1)
        trues_test = np.concatenate(trues_test, axis=0).reshape(-1)
        results = {"test/loss": sum(test_losses) / len(test_losses),
                   "test/accuracy": metrics.accuracy_score(trues_test > 0.5, outs_test > 0.5),
                   "test/precision": metrics.precision_score(trues_test > 0.5, outs_test > 0.5),
                   "test/recall": metrics.recall_score(trues_test > 0.5, outs_test > 0.5),
                   "test/f1": metrics.f1_score(trues_test > 0.5, outs_test > 0.5)}
        wandb.log(results)

        # all_losses.append(sum(test_losses) / len(test_losses))
        # all_accuracies.append(sum(test_accuracies) / len(test_accuracies))
        # all_precisions.append(sum(test_precisions) / len(test_precisions))
        # all_recalls.append(sum(test_recalls) / len(test_recalls))
        # all_f1s.append(sum(test_f1s) / len(test_f1s))

        # with open("results.txt", "a") as outfile:
        #     outfile.write(str(results))

        run.finish()

    # print("OVERALL LOSS: {}+-{}".format((sum(all_losses) / len(all_losses)), np.var(all_losses)))
    # print("OVERALL ACCURACY: {}+-{}".format((sum(all_accuracies) / len(all_accuracies)), np.var(all_accuracies)))
    # print("OVERALL PRECISION: {}+-{}".format((sum(all_precisions) / len(all_precisions)), np.var(all_precisions)))
    # print("OVERALL RECALL: {}+-{}".format((sum(all_recalls) / len(all_recalls)), np.var(all_recalls)))
    # print("OVERALL F1: {}+-{}".format((sum(all_f1s) / len(all_f1s)), np.var(all_f1s)))
