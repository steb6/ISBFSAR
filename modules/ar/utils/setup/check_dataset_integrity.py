import os
from tqdm import tqdm

if __name__ == "__main__":
    # root = "D:\\datasets\\NTURGBD_FINAL_IMAGES"
    root = "../NTURGBD_FINAL_IMAGES"
    classes = os.listdir(root)
    for c in tqdm(classes):
        samples = os.listdir(os.path.join(root, c))
        for sample in samples:
            sample_path = os.path.join(root, c, sample)
            if len(os.listdir(sample_path)) != 16:
                print("Removing directory with", len(os.listdir(sample_path)), "images instead of 16:",
                      os.path.join(root, c, sample))
                os.rmdir(sample_path)
