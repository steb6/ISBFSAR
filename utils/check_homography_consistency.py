import cv2
import numpy as np
from modules.hpe.utils.misc import homography, get_augmentations, image_transformation, ImageTransformer
from utils.params import RealSenseIntrinsics
import torch

if __name__ == "__main__":
    num_aug = 5
    x1 = 0
    x2 = 359
    y1 = 112
    y2 = 478
    inp = cv2.imread('assets/input.jpg')

    cam_config = RealSenseIntrinsics()
    K = np.zeros((3, 3), np.float32)
    K[0][0] = cam_config.fx
    K[0][2] = cam_config.ppx
    K[1][1] = cam_config.fy
    K[1][2] = cam_config.ppy
    K[2][2] = 1

    new_K, homo_inv = homography(x1, x2, y1, y2, K, 256)

    # Augmentation TODO ADD GAMMA DECODING
    aug_should_flip, aug_rotflipmat, aug_gammas, aug_scales = get_augmentations(num_aug)
    new_K = np.tile(new_K, (num_aug, 1, 1))
    for k in range(num_aug):
        new_K[k, :2, :2] *= aug_scales[k]
    homo_inv = aug_rotflipmat @ np.tile(homo_inv[0], (num_aug, 1, 1))

    # Homography
    H = K @ np.linalg.inv(new_K @ homo_inv)

    # TODO NEW
    import time
    while True:
        start = time.time()
        image_transformer = ImageTransformer(num_aug, 480, 640).cuda()
        inp_ = torch.IntTensor(inp).cuda()
        H_ = torch.FloatTensor(H).cuda()
        bbone_in = image_transformer(inp_, H_, 256)
        bbone_in = bbone_in.detach().float().cpu().numpy()
        print(time.time() - start)
    # TODO OLD
    # bbone_in = image_transformation(inp, H, 256)
    # TODO END

    bbone_in_ = (bbone_in / 255.0).astype(np.float32)

    inp = cv2.rectangle(inp, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("input", inp)
    cv2.waitKey(0)
    for elem in bbone_in:
        cv2.imshow("input", elem.astype(np.uint8))
        cv2.waitKey(0)
