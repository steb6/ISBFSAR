import numpy as np
import torch

BATCH_SIZE = 1  # NOTE" the input has always batch size = 1, this is the batch size of the transformation matrix


class ImageTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1280, 288)

    def forward(self, x):
        x = x.reshape(1, 8, 8, 1280)
        y = self.layer(x)
        return y


if __name__ == '__main__':
    inp_ = torch.FloatTensor(np.random.random(81920)).cuda()

    model = ImageTransformer()
    head_weights = np.load('modules/hpe/modules/numpy/head_weights.npy')
    head_bias = np.load('modules/hpe/modules/numpy/head_bias.npy')
    model.layer.weight.data = torch.FloatTensor(head_weights.squeeze().T)
    model.layer.bias.data = torch.FloatTensor(head_bias)
    model.cuda()

    _ = model(inp_)

    torch.onnx.export(model, (inp_,), 'modules/hpe/modules/onnxs/heads1.onnx',
                      input_names=['input'], output_names=['output'],
                      opset_version=9, verbose=True)
