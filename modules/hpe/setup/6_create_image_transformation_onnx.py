import numpy as np
import torch


BATCH_SIZE = 5  # NOTE" the input has always batch size = 1, this is the batch size of the transformation matrix

# python C:\Users\sberti\AppData\Local\mambaforge\envs\robotologyenv\Scripts\polygraphy run modules/hpe/modules/onnxs/image_transformation1.onnx --trt --onnxrt --trt-outputs mark all --onnx-outputs mark all --fail-fast


class ImageTransformer(torch.nn.Module):
    def __init__(self, n_batch, h, w):
        super().__init__()
        self.w = w
        self.h = h
        self.batch_offset = self.w * self.h
        self.n_batch = n_batch
        self.x = torch.linspace(0, w - 1, w)[None].repeat(h, 1).reshape(-1).int()[None].repeat(self.n_batch, 1).cuda()
        self.y = torch.linspace(0, h - 1, h).repeat_interleave(w).int()[None].repeat(self.n_batch, 1).cuda()
        self.in_idx = self.x + self.y * w
        self.offsets = torch.full(self.in_idx.shape, self.batch_offset).cuda() * torch.linspace(0, self.n_batch - 1, self.n_batch)[..., None].int().cuda()
        self.in_idx = self.in_idx + self.offsets

    def forward(self, img, transform):

        # Make transform matrix a normalized vector
        transform = transform / transform[:, 2:3, 2:3]  # last element must be 1
        transform = torch.reshape(transform, [self.n_batch, 9])[:, :8]
        transform = transform[..., None]

        # Prepare indices for input
        k = transform[:, 6] * self.x + transform[:, 7] * self.y + 1
        xi = transform[:, 0] * self.x + transform[:, 1] * self.y + transform[:, 2]
        xi = (xi / k).int()
        yi = transform[:, 3] * self.x + transform[:, 4] * self.y + transform[:, 5]
        yi = (yi / k).int()
        out_idx = xi + yi * self.w
        out_idx = out_idx + self.offsets  # OK

        # Flat images
        out_flat = img.clone()[None, ...].repeat(self.n_batch, 1, 1, 1).reshape(-1, 3)
        in_flat = img.clone()[None, ...].repeat(self.n_batch, 1, 1, 1).reshape(-1, 3)

        # Fix bad indices
        xi_outside = torch.logical_or(xi < 0, xi >= self.w)
        yi_outside = torch.logical_or(yi < 0, yi >= self.h)
        bad_idx = torch.logical_or(xi_outside, yi_outside)
        out_idx[bad_idx] = -1

        # Transform
        out_flat[self.in_idx.reshape(-1)] = torch.where((out_idx.reshape(-1) != -1).unsqueeze(-1).repeat(1, 3),
                                                        in_flat[out_idx.reshape(-1)], torch.zeros_like(in_flat))

        # Reshape for output
        out_flat = out_flat.reshape(self.n_batch, self.h, self.w, 3)
        out_flat = out_flat[:, :256, :256, :]
        return out_flat


if __name__ == '__main__':
    inp_ = torch.IntTensor(np.random.random((480, 640, 3))).cuda()

    H_ = torch.FloatTensor(np.random.random((BATCH_SIZE, 3, 3))).cuda()

    model = ImageTransformer(BATCH_SIZE, 480, 640)
    model.cuda()
    torch.onnx.export(model, (inp_, H_), 'modules/hpe/modules/onnxs/image_transformation{}.onnx'.format(BATCH_SIZE),
                      input_names=['frame', 'H'], output_names=['images'],
                      opset_version=11)
