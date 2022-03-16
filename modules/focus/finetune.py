import torch
from modules.focus.utils.SCUTDataset import SCUTDataset
from modules.focus.utils.engine import train_one_epoch, evaluate
from modules.focus.utils.misc import get_model, get_transform
import modules.focus.utils.utils as utils


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
    dataset = SCUTDataset('D:/SCUT', get_transform(train=True))
    dataset_test = SCUTDataset('D:/SCUT', get_transform(train=False))

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

    # TODO TEST ON FIXED SAMPLE
    # sample = data_loader_test.__iter__().__next__()
    # sample = tuple(sample[0][0].cuda())

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
    num_epochs = 100

    for epoch in range(num_epochs):
        # # TODO VISUAL RESULT
        # model.eval()
        # res = model(sample)
        # model.train()
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        print("STARTING VALIDATION")
        evaluate(model, data_loader_test, device=device)
        # Save model
        torch.save(model.state_dict(), "epoch_{}.pth".format(epoch))
        print("Model saved into ", "epoch_{}".format(epoch))

    print("That's it!")


if __name__ == "__main__":
    train()
