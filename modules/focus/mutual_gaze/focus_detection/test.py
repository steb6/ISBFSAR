import torch.optim
from torchvision import transforms
from modules.focus.mutual_gaze.focus_detection.utils.MARIADataset import MARIAData
from modules.focus.mutual_gaze.focus_detection.utils.model import MutualGazeDetectorHeads as Model
from tqdm import tqdm
from sklearn import metrics
import cv2


# ckpt_path = "modules/focus/mutual_gaze/focus_detection/checkpoints/f1_0.82_loss_7.24_HEADS_maria_aug.pth"
# {'test/accuracy': '0.91', 'test/precision': '0.88', 'test/recall': '0.96', 'test/f1': '0.92'}

ckpt_path = "modules/focus/mutual_gaze/focus_detection/checkpoints/f1_0.96_loss_0.14_HEADS_my_aug.pth"
# {'test/accuracy': '0.88', 'test/precision': '0.90', 'test/recall': '0.85', 'test/f1': '0.87'}

if __name__ == "__main__":
    test_data = MARIAData("D:/datasets/focus_dataset_heads", mode="test", split_number=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=2, shuffle=True)

    model = Model()
    model.load_state_dict(torch.load(ckpt_path))
    model.cuda()
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Test set length: {:.2f}".format(len(test_loader)))
    print("Train set watching: {:.2f}, not watching: {:.2f}".format(test_data.n_watch, test_data.n_not_watch))

    # Test
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    outs = []
    imgs = []
    ys = []

    for img, y in tqdm(test_loader):
        x = img.permute(0, 3, 1, 2)
        x = x / 255.
        x = normalize(x)
        x = x.cuda()
        y = y.cuda()

        out = model(x)
        outs.extend(out.cpu().detach().numpy()[:, 0].tolist())
        ys.extend(y.cpu().detach().numpy().tolist())
        imgs.extend([_ for _ in img.detach().cpu().numpy()])

        test_accuracies.append(metrics.accuracy_score(y.detach().cpu() > 0.5, out.detach().cpu() > 0.5))
        test_precisions.append(metrics.precision_score(y.detach().cpu() > 0.5, out.detach().cpu() > 0.5))
        test_recalls.append(metrics.recall_score(y.detach().cpu() > 0.5, out.detach().cpu() > 0.5))
        test_f1s.append(metrics.f1_score(y.detach().cpu() > 0.5, out.detach().cpu() > 0.5))

    print({"test/accuracy": "{:.2f}".format(sum(test_accuracies) / len(test_accuracies)),
           "test/precision": "{:.2f}".format(sum(test_precisions) / len(test_precisions)),
           "test/recall": "{:.2f}".format(sum(test_recalls) / len(test_recalls)),
           "test/f1": "{:.2f}".format(sum(test_f1s) / len(test_f1s))})

    for img, label, true in zip(imgs, outs, ys):
        print("True: {}, Pred: {}".format(true, label))
        cv2.imshow("img", img)
        cv2.waitKey(0)
