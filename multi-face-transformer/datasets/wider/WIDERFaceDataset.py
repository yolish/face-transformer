# Some processing functionscopied from: tiny-face-pytorch and adopted for repo needs
import logging
import numpy as np
from PIL import Image
from torch.utils.data import dataset
from torchvision import transforms
import json
from os.path import join
import torch
from util import box_ops


class WIDERFaceDataset(dataset.Dataset):
    """The WIDERFace dataset is generated using MATLAB,
    so a lot of small housekeeping elements have been added
    to take care of the indexing discrepancies."""
    def __init__(self, dataset_path, dataset_file, split, img_transforms=None,
                debug=False, format="box"):
        super().__init__()

        self.split = split
        self.dataset_path = dataset_path
        self.data = self.read_dataset_file(dataset_file)

        logging.info("Dataset loaded")
        logging.info("{} samples in the {} dataset".format(len(self.data),
                                                      self.split))

        self.transforms = img_transforms
        self.debug = debug
        self.format = format
        self.img_paths = []
        for datum in self.data:
            self.img_paths.append(datum.get('img_path'))


    def read_dataset_file(self, file_path):
        """Load the dataset from the text file."""

        if self.split in ("train", "val"):
            img_dir = join(join(self.dataset_path, "WIDER_{}".format(self.split)), "images")
            lines = open(file_path).readlines()

            data = []
            for line in lines:
                json_file = line.strip()
                with open(join(self.dataset_path, json_file)) as f:
                    d = json.load(f)
                    img_path = join(img_dir, d.get("image_path"))
                    bboxes = d.get("bboxes")
                    # format is x0,y0,x1,y1
                    bboxes = np.array([np.array(box[:4]) for box in bboxes])
                    # Remove invalid bboxes where w or h are 0
                    invalid = np.where(np.logical_or(bboxes[:, 2] < bboxes[:, 0] ,
                                                     bboxes[:, 3] < bboxes[:, 1] ))
                    bboxes = np.delete(bboxes, invalid, 0)
                    datum = {
                        "img_path": img_path,
                        "bboxes": bboxes
                    }
                    data.append(datum)

        elif self.split == "test":
            # TODO Handle later
            data = open(file_path).readlines()
            data = [{'img_path': x.strip()} for x in data]
        return data

    def get_all_bboxes(self):
        bboxes = np.empty((0, 4))
        for datum in self.data:
            bboxes = np.vstack((bboxes, datum['bboxes']))

        return bboxes

    def get_img_path(self, index):
        datum = self.data[index]
        return datum.get('img_path')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]

        img = Image.open(datum.get('img_path')).convert('RGB')
        orig_w, orig_h = img.size # (PIL image)
        bboxes = datum['bboxes']

        if self.debug:
            if bboxes.shape[0] == 0:
                logging.warning("Image at {} has zero bounding boxes".format(datum.get('img_path')))
            logging.info("Loading image: {}, dataset index: {}".format(datum.get('img_path'), index))

        target = {}
        target["labels"] = torch.zeros(bboxes.shape[0]).to(dtype=torch.int64)
        target["boxes"] = torch.from_numpy(bboxes).to(dtype=torch.float) # Face = 0, No object = 1
        target["size"] = torch.as_tensor([int(orig_h), int(orig_w)])
        # convert everything to tensors
        if self.transforms is not None:
            # if img is a byte or uint8 array, it will convert from 0-255 to 0-1
            # this converts from (HxWxC) to (CxHxW) as well
            img, target = self.transforms(img, target)

        mask = torch.zeros(img.shape)
        target["masks"] = mask
        h, w = img.shape[1:]
        target["size"] = torch.as_tensor([int(h), int(w)])


        '''
        mask = torch.zeros(img.shape)

        target = {}
        target["boxes"] = torch.from_numpy(bboxes).to(dtype=torch.float)
        target["labels"] = torch.ones(bboxes.shape[0]).to(dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(orig_h), int(orig_w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["masks"] = mask
        '''
        return img,  target
