from modules.ar.utils.dataloader import EpisodicLoader
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig
import numpy as np
import torch
from itertools import combinations
import cv2
import random

device = TRXConfig().device

# REPRODUCIBILITY
# torch.manual_seed(0)
# random.seed(1)


if __name__ == "__main__":
    args = TRXConfig()

    # LOAD LIST OF TEST CLASSES
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

    # LOAD MODEL
    model = TRXOS(args, add_hook=True).to(args.device)
    # TODO START RENAME
    params = torch.load('modules/ar/modules/raws/hybrid/2500.pth')['model_state_dict']
    aux = {}
    for param in params:
        data = params[param]
        if 'module' in param:
            aux[param.replace('.module', '')] = data
        else:
            aux[param] = data
    # TODO END
    model.load_state_dict(aux)
    model.eval()

    # LOAD DATASET
    train_data = EpisodicLoader(args.data_path, k=args.way, n_task=100, input_type=args.input_type, )
    train_data.classes = test_classes
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=0, shuffle=True)

    # LOSSES
    os_loss_fn = torch.nn.BCELoss()
    fs_loss_fn = torch.nn.CrossEntropyLoss()
    test_loss_fn = torch.nn.MSELoss()

    # PREPARE SAMPLE
    for elem in train_loader:

        # Extract from dict, convert, move to GPU
        support_set = [elem['support_set'][t].float().to(device) for t in elem['support_set'].keys()]
        target_set = [elem['target_set'][t].float().to(device) for t in elem['target_set'].keys()]
        unknown_set = [elem['unknown_set'][t].float().to(device) for t in elem['unknown_set'].keys()]

        support_labels = torch.arange(args.way).repeat(1).reshape(1, args.way).to(device).int()
        target = (elem['support_classes'] == elem['target_class'][..., None]).float().to(device)

        # INFERENCE
        gradients_trg = None
        gradients_ss = None
        activations_trg = None
        activations_ss = None
        heatmap_ss = None
        heatmap_trg = None
        model.zero_grad()
        model.features_extractor["rgb"].activations = []
        model.features_extractor["rgb"].gradients = []
        out = model(support_set, support_labels, target_set)
        fs_pred = out['logits']
        if torch.argmax(fs_pred) == torch.argmax(target):
            print("CORRECT!")
        else:
            print("FAIL!")
        print("Predicted:", fs_pred.detach().cpu().numpy())
        print("Real:", target.detach().cpu().numpy())

        # COMPUTE GRADIENTS
        # true_index = (target == 1).nonzero()[0][1]  # TRUE
        true_index = torch.argmax(fs_pred)  # PREDICTED
        # model.zero_grad()
        # (fs_pred[:, true_index]).backward()
        # known_fs_loss = fs_loss_fn(fs_pred[:, true_index][None], target[:, true_index][None])  # JUST TRUE
        target.requires_grad = True
        # one_hot = torch.sum(target * fs_pred)
        # known_fs_loss = fs_loss_fn(fs_pred, target)  # WHOLE
        # known_fs_loss = test_loss_fn(fs_pred[:, true_index], torch.FloatTensor([0]).cuda())  # WHOLE
        model.zero_grad()
        fs_pred[:, true_index].backward()

        # GET GRADIENTS, SCORES AND ACTIVATIONS
        gradients_trg, gradients_ss = model.features_extractor["rgb"].get_activations_gradient()
        activations_ss, activations_trg = model.features_extractor["rgb"].activations

        # TRANSFORM DATA INTO IMAGES
        support_set = (support_set[0].permute(0, 1, 2, 4, 5, 3) - torch.FloatTensor([0.485, 0.456, 0.406]).cuda()) / torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
        support_set = support_set.detach().cpu().numpy()
        support_set = (support_set * 255).astype(int)
        support_set = support_set[0].swapaxes(0, 1).reshape(8, 224*5, 224, 3).swapaxes(0, 1).reshape(5*224, 8*224, 3)

        target_set = (target_set[0].permute(0, 1, 3, 4, 2) - torch.FloatTensor([0.485, 0.456, 0.406]).cuda()) / torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
        target_set = target_set.detach().cpu().numpy()
        target_set = (target_set * 255).astype(int)
        target_set = target_set[0].swapaxes(0, 1).reshape(224, -1, 3)

        # COMPUTE BEST SCORES
        frame_idxs = [i for i in range(8)]
        frame_combinations = combinations(frame_idxs, 2)
        tuples = [torch.tensor(comb).to(args.device) for comb in frame_combinations]

        scores = model.transformers[0].scores[true_index][0][0].detach().cpu().numpy()

        best_ss = tuples[np.argmax(np.sum(scores, axis=1))]
        a = tuples[np.argmax(scores) % 28]
        b = tuples[int(np.floor(np.argmax(scores) / 28))]

        cv2.imwrite("scores.jpg", (scores / np.max(scores))*255)
        print("Best pair score:", np.max(scores))
        print("Best query pairs:", a.detach().cpu().numpy())
        print("Best class pairs", b.detach().cpu().numpy())
        print("Best ss:", best_ss)

        # CREATE HEATMAP IMAGES
        gradients_ss = gradients_ss.mean([2, 3])
        gradients_trg = gradients_trg.mean([2, 3])

        # TODO JUST BEST GRADIENTS
        # best_gradients_ss = (-gradients_ss).argmax(dim=1)
        # best_gradients_trg = (-gradients_trg).argmax(dim=1)
        # best_activations = 3
        #
        # best_ss = []
        # best_trg = []
        # for i in range(40):
        #     best_ss.append(activations_ss[i, best_gradients_ss[i], ...])
        # for i in range(8):
        #     best_trg.append(activations_trg[i, best_gradients_trg[i], ...])
        # heatmap_ss = torch.stack(best_ss).cpu().numpy()
        # heatmap_trg = torch.stack(best_trg).cpu().numpy()

        # TODO best is + and -
        for i in range(gradients_ss.shape[1]):
            activations_ss[:, i, :, :] *= gradients_ss[:, i][..., None, None]
            activations_trg[:, i, :, :] *= gradients_trg[:, i][..., None, None]

            # activations_ss[:, i, :, :] *= activations_ss[:, i, :, :].mean(dim=(1, 2))[..., None, None]
            # activations_trg[:, i, :, :] *= activations_trg[:, i, :, :].mean(dim=(1, 2))[..., None, None]

            # for f in range(8):
            #     print(gradients_trg[f, i])
            #     act = activations_trg.reshape(8, 2048, 7, 7)[f][i].detach().cpu().numpy()
            #     act = np.maximum(act, 0)
            #     act = (act / np.max(act)) * 255
            #     cv2.imshow(f"a {i}, img {f}", cv2.resize(act.astype(np.uint8), (224, 224)))
            #     cv2.waitKey(0)
            #     cv2.imwrite(f"a{i}-img{f}.png", cv2.resize(act.astype(np.uint8), (224, 224)))
        heatmap_ss = torch.mean(activations_ss, 1).detach().cpu().numpy()
        heatmap_trg = torch.mean(activations_trg, 1).detach().cpu().numpy()

        # TODO END

        size = activations_ss.shape[-1]
        # heatmap_ss[..., :2, :2] = 0  # TODO REMOVE DEBUG
        heatmap_ss = np.maximum(heatmap_ss, 0)
        heatmap_ss /= np.max(heatmap_ss, axis=(1, 2), keepdims=True)
        heatmap_ss = heatmap_ss.reshape(5, 8, size, size).swapaxes(0, 1).reshape(8, 5*size, size).swapaxes(0, 1).reshape(5*size, size*8)

        # heatmap_trg[..., :2, :2] = 0  # TODO REMOVE DEBUG
        heatmap_trg = np.maximum(heatmap_trg, 0)
        heatmap_trg /= np.max(heatmap_trg, axis=(1, 2), keepdims=True)
        heatmap_trg = heatmap_trg.reshape(1, 8, size, size).swapaxes(0, 1).reshape(8, 1*size, size).swapaxes(0, 1).reshape(1*size, size*8)

        heats = []
        def save_heatmap_img(img, heatmap, name):
            if name == "ss":
                heatmap = heatmap[int(heatmap.shape[0] / 5) * true_index.item():int(heatmap.shape[0] / 5) * (
                            true_index + 1).item(), ...]
                img = img[int(img.shape[0] / 5) * true_index.item():int(img.shape[0] / 5) * (
                            true_index + 1).item(), ...]
            heats.append(heatmap)
            heatmap *= 255
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(f"{name}_heatmap.jpg", heatmap)
            heatmap = np.uint8(heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img
            cv2.imwrite(f'{name}.jpg', superimposed_img)


        save_heatmap_img(support_set, heatmap_ss, "ss")
        save_heatmap_img(target_set, heatmap_trg, "trg")

        # TODO VISUALIZE FILTER

        # cannot easily visualize filters lower down
        # from keras.applications.vgg16 import VGG16
        # from matplotlib import pyplot
        #
        # # load the model
        # model = model.features_extractor
        # # retrieve weights from the second hidden layer
        # filters = np.array([model.features_extractor.layer1[0].conv1.weight.data.cpu().numpy().squeeze()])
        # # normalize filter values to 0-1 so we can visualize them
        # f_min, f_max = filters.min(), filters.max()
        # filters = (filters - f_min) / (f_max - f_min)
        # # plot first few filters
        # n_filters, ix = 6, 1
        # for i in range(n_filters):
        #     # get the filter
        #     f = filters[:, :, :, i]
        #     # plot each channel separately
        #     for j in range(3):
        #         # specify subplot and turn of axis
        #         ax = pyplot.subplot(n_filters, 3, ix)
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         # plot filter channel in grayscale
        #         pyplot.imshow(f[:, :, j], cmap='gray')
        #         ix += 1
        # # show the figure
        # pyplot.show()

        input()
