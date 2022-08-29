import pickle
import os
import shutil
import sys
sys.path.append("modules/hpe/assets/pytorchYOLOv4")
from modules.hpe.assets.pytorchYOLOv4.models import Yolov4
import cv2
from tqdm import tqdm
from modules.hpe.utils.misc import postprocess_yolo_output
from utils.params import MetrabsTRTConfig
import numpy as np
from utils.params import ubuntu
import torch

# sent 51,663,880,598 bytes  received 27,960,450 bytes  1,401,375.60 bytes/sec
# total size is 51,578,421,405  speedup is 1.00

# missing = ["blow_nose", "falling", "nausea_or_vomiting_condition", "sneeze-cough", "staggering", "stretch_oneself",
#            "touch_back_(backache)", "touch_chest_(stomachache-heart_pain)", "touch_head_(headache)",
#            "touch_neck_(neckache)", "use_a_fan_(with_hand_or_paper)-feeling_warm", "yawn"]

in_dataset_path = "D:\\datasets\\useless\\nturgbd" if not ubuntu else "../nturgbd"
out_dataset_path = "D:\\datasets\\NTURGBD_FINAL_IMAGES_no_pad" if not ubuntu else "../NTURGBD_FINAL_IMAGES_no_pad"
classes_path = "assets/nturgbd_classes.txt"
n = 8

# it reached 35978

VIS_DEBUG = False

if __name__ == "__main__":

    # raise Exception("REMOVE THIS LINE TO ERASE CURRENT DATASET")  # TODO ADD

    skeleton = 'smpl+head_30'
    with open('assets/skeleton_types.pkl', "rb") as input_file:
        skeleton_types = pickle.load(input_file)
    edges = skeleton_types[skeleton]['edges']

    model_config = MetrabsTRTConfig()

    # TODO LOAD YOLO
    WEIGHT_FILE = "modules/hpe/modules/raws/yolov4.pth" if not ubuntu else "yolov4.pth"
    N_CLASSES = 80

    model = Yolov4(n_classes=N_CLASSES, inference=True)
    pretrained_dict = torch.load(WEIGHT_FILE, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.eval()
    model.cuda()
    # TODO END

    # Count total number of files (we remove 10 classes over 60 because those involves two person)
    total = int(sum([len(files) if '_s' in r else 0 for r, d, files in os.walk(in_dataset_path)]) * (1 - 26/120))

    # Get conversion class id -> class label
    with open(classes_path, "r", encoding='utf-8') as f:
        classes = f.readlines()
    class_dict = {}
    for c in classes:
        index, name, _ = c.split(".")
        name = name.strip().replace(" ", "_").replace("/", "-").replace("’", "")
        class_dict[index] = name

    # # Create output directories
    os.mkdir(out_dataset_path)
    for i, value in enumerate(list(class_dict.values())):
        if 0 <= i <= 49-1 or 61-1 <= i <= 105-1:
            if os.path.exists(os.path.join(out_dataset_path, value)):
                shutil.rmtree(os.path.join(out_dataset_path, value))
            os.mkdir(os.path.join(out_dataset_path, value))

    curl = 0  # TODO REMOVE DEBUG
    # Iterate all videos
    with tqdm(total=total) as progress_bar:
        curl += 1  # TODO REMOVE DEBUG
        for root, dirs, files in os.walk(in_dataset_path):

            # if '_s' in root:  # BEFORE it was `not in root`
            #     continue

            for file in files:
                # Retrieve class name (between A and _ es 'S001C001P001R001A001_rgb.avi'
                class_id = int(file.split("A")[1].split("_")[0])  # take the integer of the class
                class_id = "A" + str(class_id)
                class_name = class_dict[class_id]

                # if class_name not in missing:
                #     continue

                # Skip if two person are involved
                if 50-1 <= list(class_dict.keys()).index(class_id) <= 60-1 or 106 - 1 <= list(class_dict.keys()).index(class_id) <= 120 - 1:
                    continue

                # Read video
                full = os.path.join(root, file)
                video = cv2.VideoCapture(full)
                frames = []
                ret, frame = video.read()
                while ret:
                    frames.append(frame)
                    ret, frame = video.read()
                if len(frames) < n:
                    continue

                # Select just n frames
                n_frames = len(frames) - (len(frames) % n)
                if n_frames == 0:
                    continue
                indices = list(range(0, n_frames, int(n_frames / n)))
                frames = [frames[i] for i in indices]

                # Iterate over all frames
                good = True
                filtered_frames = []
                for i, frame in enumerate(frames):

                    frame = frame[:, 240:-240, :]

                    # TODO START, EXTRACT HUMAN AND RESIZE
                    # Preprocess for yolo
                    square_img = cv2.resize(frame, (256, 256), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
                    square_img = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
                    square_img = np.transpose(square_img, (2, 0, 1)).astype(np.float32)
                    square_img = np.expand_dims(square_img, axis=0)
                    square_img = square_img / 255.0

                    # Yolo
                    square_img = torch.FloatTensor(square_img).cuda()
                    outputs = model(square_img)
                    boxes, confidences = outputs[0].reshape(1, 4032, 1, 4), outputs[1].reshape(1, 4032, 80)
                    boxes, confidences = boxes.detach().cpu().numpy(), confidences.detach().cpu().numpy()
                    bboxes_batch = postprocess_yolo_output(boxes, confidences, model_config.yolo_thresh,
                                                           model_config.nms_thresh)

                    # Get only the bounding box with the human with highest probability
                    box = bboxes_batch[0]  # Remove batch dimension
                    humans = []
                    for e in box:  # For each object in the image
                        if e[5] == 0:  # If it is a human
                            humans.append(e)
                    if len(humans) > 0:
                        humans.sort(key=lambda x: x[4], reverse=True)  # Sort with decreasing probability
                        human = humans[0]
                    else:
                        good = False

                    # Preprocess for BackBone
                    x1 = int(human[0] * frame.shape[1]) if int(human[0] * frame.shape[1]) > 0 else 0
                    y1 = int(human[1] * frame.shape[0]) if int(human[1] * frame.shape[0]) > 0 else 0
                    x2 = int(human[2] * frame.shape[1]) if int(human[2] * frame.shape[1]) > 0 else 0
                    y2 = int(human[3] * frame.shape[0]) if int(human[3] * frame.shape[0]) > 0 else 0

                    # TODO START NEW
                    xm = int((x1 + x2) / 2)
                    ym = int((y1 + y2) / 2)
                    l = max(xm - x1, ym - y1)
                    frame = frame[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
                    # TODO START OLD
                    #
                    # frame = frame[y1:y2, x1:x2]
                    # frame = cv2.copyMakeBorder(frame,
                    #                            int((frame.shape[1] - frame.shape[0]) / 2) if frame.shape[1] > frame.shape[0] else 0,
                    #                            int((frame.shape[1] - frame.shape[0]) / 2) if frame.shape[1] > frame.shape[0] else 0,
                    #                            int((frame.shape[0] - frame.shape[1]) / 2) if frame.shape[0] > frame.shape[1] else 0,
                    #                            int((frame.shape[0] - frame.shape[1]) / 2) if frame.shape[0] > frame.shape[1] else 0,
                    #                            cv2.BORDER_CONSTANT)
                    # TODO END

                    frame = cv2.resize(frame, (224, 224))
                    # cv2.imshow("frame", frame.astype(np.uint8))  # TODO VISUALIZE DEBUG
                    # cv2.waitKey(0)  # TODO VISUALIZE DEBUG
                    filtered_frames.append(frame)

                if good:
                    # Check if output path already exists
                    output_path = os.path.join(out_dataset_path, class_name)
                    offset = sum([len(d) for r, d, files in os.walk(output_path)])
                    output_path = os.path.join(output_path, str(offset))
                    os.mkdir(output_path)

                    for i, frame in enumerate(filtered_frames):
                        cv2.imwrite(os.path.join(output_path, f"{i}.png"), frame)

                progress_bar.update()
