import torch
import sys
import os
try:
    sys.path.append(os.path.join("modules", "hpe", "assets", "pytorchYOLOv4"))
    from modules.hpe.assets.pytorchYOLOv4.models import Yolov4
except ImportError as e:
    print(e)
    print("Clone https://github.com/Tianxiaomo/pytorch-YOLOv4 inside modules/hpe/assets first!")
    print("Then remove the '-' character from Pytorch-YOLOv4")
    print("Launch from main directory")
    exit(-1)

if not os.path.exists('modules/hpe/modules/raws/yolov4.pth'):
    print("Download the weights from https://drive.google.com/file/d/1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ/view")
    print("Or yolov4.pth from the repository page https://github.com/Tianxiaomo/pytorch-YOLOv4")
    print("And move it inside modules/hpe/modules/raws")
    exit(-1)


BATCH_SIZE = 1
H = 256
W = 256
N_CLASSES = 80
WEIGHT_FILE = "modules/hpe/modules/raws/yolov4.pth"
OUTPUT_FILE = "modules/hpe/modules/onnxs/yolo.onnx"


def rewrite(mod, weight_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.load(weight_path, map_location=device)

    weight_dict = {}
    for key, val in weight.items():
        if 'neek' in key:
            key = key.replace("neek", "neck")
        weight_dict.update({key: val})
    mod.load_state_dict(weight_dict)
    torch.save(mod.state_dict(), output_path)


if __name__ == "__main__":
    model = Yolov4(n_classes=N_CLASSES, inference=True)

    rewrite(model, WEIGHT_FILE, WEIGHT_FILE)

    pretrained_dict = torch.load(WEIGHT_FILE, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    x = torch.randn((BATCH_SIZE, 3, 256, 256), requires_grad=True)
    # Export the model
    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      OUTPUT_FILE,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names)

    print('Onnx model exporting done')
