import math
import sys
import torch
from modules.focus.mutual_gaze.head_detection.utils.SCUTDataset import SCUTDataset
import time
from modules.focus.mutual_gaze.head_detection.utils.misc import get_model, get_transform
import modules.focus.mutual_gaze.head_detection.utils.utils as utils


def test():
    model = get_model()
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    dataset = SCUTDataset('D:/SCUT', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
     dataset, batch_size=2, shuffle=True, num_workers=0,
     collate_fn=utils.collate_fn)
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)   # Returns losses and detections
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)           # Returns predictions
    return predictions


###########################
# TRAINING AND VALIDATION #
###########################
def train():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = SCUTDataset('D:/datasets/SCUT', get_transform(train=True))
    dataset_test = SCUTDataset('D:/datasets/SCUT', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    n_test = int(len(dataset) * 0.2)
    dataset = torch.utils.data.Subset(dataset, indices[:-n_test])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-n_test:])
    # dataset = torch.utils.data.Subset(dataset, indices[:5])  # TODO REMOVE DEBUG
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[:5])  # TODO REMOVE DEBUG

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model()

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 15
    print_freq = 10
    scaler = None

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # TODO START
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # TODO END
        # update the learning rate
        lr_scheduler.step()
        # Evaluate
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            # coco_evaluator.update(res)  # TODO REMOVED
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        # coco_evaluator.synchronize_between_processes()  # TODO REMOVED

        # accumulate predictions from all images
        # coco_evaluator.accumulate()  # TODO REMOVED
        # coco_evaluator.summarize()  # TODO REMOVED
        torch.set_num_threads(n_threads)
        # Save model
        torch.save(model.state_dict(), "epoch_{}.pth".format(epoch))
        print("Model saved into ", "epoch_{}".format(epoch))

    print("That's it!")


if __name__ == "__main__":
    train()
