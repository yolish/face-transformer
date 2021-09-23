
from os.path import join, basename
import numpy as np
import json
import cv2
import argparse
from pathlib import Path
import pandas as pd


def crop_dataset(dataset_path, split, out_path):
    """Load the dataset from the text file."""

    source_imgs = []
    cropped_boxes = []
    file_path = join(dataset_path, "annotations/WIDER_{}_annotations.txt".format(split))
    img_dir = join(join(dataset_path, "WIDER_{}".format(split)), "images")
    lines = open(file_path).readlines()
    for line in lines:
        json_file = line.strip()
        with open(join(dataset_path, json_file)) as f:
            d = json.load(f)

            img_path = join(img_dir, d.get("image_path"))
            bboxes = d.get("bboxes")
            # format is x0,y0,x1,y1
            bboxes = np.array([np.array(box[:4]).astype(np.int) for box in bboxes])

            # read the image
            img = cv2.imread(img_path)
            for i, box in enumerate(bboxes):
                if box[2] < box[0] or box[3] < box[1]:
                    crop_path = None
                else:
                    # crop
                    crop_path = join(out_path, basename(img_path) + "_{}.png".format(i))
                    crop = img[box[1]:(box[3] + 1), box[0]:(box[2] + 1)]
                    cv2.imwrite(crop_path, crop)
                source_imgs.append(img_path)
                cropped_boxes.append(crop_path)
    df = pd.DataFrame({"src_img":source_imgs, "cropped_box_img":cropped_boxes})
    df.to_csv(join(out_path, "cropped_wider_{}.csv".format(split)), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='path to dataset')
    parser.add_argument('output_path', type=str, help='path to outputs')
    args = parser.parse_args()
    if args.output_path:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
    crop_dataset(args.dataset_path, 'train', args.output_path)
    crop_dataset(args.dataset_path, 'val', args.output_path)



