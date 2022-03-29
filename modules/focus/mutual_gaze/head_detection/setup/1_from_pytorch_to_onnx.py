import torch

from modules.focus.head_detection.utils.misc import get_model

if __name__ == "__main__":
    model = get_model()
    model.load_state_dict(torch.load('modules/focus/head_detection/modules/raw/longest.pth'))
    model.cuda()
    model.eval()

    input_names = ["img"]
    output_names = ['bbox', 'scores']

    x = torch.randn((1, 3, 480, 640), requires_grad=True).cuda()
    # Export the model
    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      'modules/focus/head_detection/modules/onnx/longest.onnx',
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names)

    print('Onnx model exporting done')