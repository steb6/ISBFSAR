import torch.optim
from torch.nn import BCELoss
import wandb
from torchvision import transforms
from modules.focus.mutual_gaze.focus_detection.MARIADataset import MARIAData
from modules.focus.mutual_gaze.focus_detection.model import MutualGazeDetector
from tqdm import tqdm

LOG_EVERY = 10

if __name__ == "__main__":
    train_data = MARIAData("D:/datasets/focus_dataset", mode="train")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=2)
    valid_data = MARIAData("D:/datasets/focus_dataset", mode="valid")
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, num_workers=2)

    model = MutualGazeDetector()
    model.cuda()
    model.train()

    loss_fn = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    run = wandb.init(project="mutual_gaze")
    wandb.watch(model, log='all', log_freq=LOG_EVERY)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Train set length: {}, valid set length: {}".format(len(train_loader), len(valid_loader)))
    print("Train set watching: {}, not watching: {}".format(train_data.n_watch, train_data.n_not_watch))
    print("Valid set watching: {}, not watching: {}".format(valid_data.n_watch, valid_data.n_not_watch))

    for _ in range(1000):
        # TRAIN
        losses = []
        accuracies = []
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):

            x = x.permute(0, 3, 1, 2)
            x = x / 255.
            x = normalize(x)
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            loss = loss_fn(out, y.float().unsqueeze(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accuracy = ((out.squeeze() > 0.5).int() == y.int()).sum().item() / torch.numel(out)
            accuracies.append(accuracy)

        wandb.log({"train/loss": sum(losses) / len(losses),
                   "train/accuracy": sum(accuracies) / len(accuracies),
                   "lr": optimizer.param_groups[0]['lr']})
        losses = []
        accuracies = []

        # EVAL
        model.eval()
        for x, y in tqdm(valid_loader):

            x = x.permute(0, 3, 1, 2)
            x = x / 255.
            x = normalize(x)
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            loss = loss_fn(out, y.float().unsqueeze(-1))

            losses.append(loss.item())
            accuracy = ((out.squeeze() > 0.5).int() == y.int()).sum().item() / torch.numel(out)
            accuracies.append(accuracy)

        wandb.log({"valid/loss": sum(losses) / len(losses),
                   "valid/accuracy": sum(accuracies) / len(accuracies),
                   "lr": optimizer.param_groups[0]['lr']})

        torch.save(model.state_dict(), "{:.2f}.pth".format(sum(accuracies) / len(accuracies)))
