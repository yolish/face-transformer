import os.path

from datasets.transforms import normalize_transform
import torch
import cv2
from util import plot_utils
import numpy as np
from os.path import join, basename, exists
from datasets.wider.WIDERFaceDataset import WIDERFaceDataset
from torch.utils.data import DataLoader
import util.misc as utils
import time
from util import box_ops


def write_results(results_file, img_name, results, box_detection_score):
    f = open(results_file, 'w')
    box_coords = (results["boxes"]).cpu().numpy()
    box_scores = results["scores"].cpu().numpy()
    confident_boxes = box_scores > box_detection_score
    selected_box_coords = box_coords[confident_boxes]
    f.write(img_name+"\n")
    f.write("{}\n".format(selected_box_coords.shape[0]))
    for i, box in enumerate(selected_box_coords):
        #top, left, width, hight, score
        f.write("{},{},{},{},{}\n".format(box[0], box[1], box[2], box[3], box_scores[i]))
    f.close()

def eval(args, model, criterion, postprocess, eval_file=None):

    # Set model
    device = torch.device(args.device)
    model.to(device)
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)

    # Set dataset and loader
    if args.eval_file is None:
        dataset_file = join(join(args.dataset_path, "annotations"), "WIDER_val_annotations.txt")
    else:
        dataset_file = eval_file

    dataset_val = WIDERFaceDataset(args.dataset_path, dataset_file, 'val', img_transforms=normalize_transform)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    f = open(join(args.output_dir, "wider_results_ckpts_{}_ds_{}.txt".format(basename(args.checkpoint_path),
                                                                             args.box_detection_score)), 'w')
    results_dir = join(args.output_dir, "wider_results")
    if not exists(results_dir):
        os.mkdir(results_dir)
    runtime = 0.0
    indices_to_plot = [100, 101, 102, 103]
    with torch.no_grad():
        model.eval()
        criterion.eval()

        for idx, (sample, targets) in enumerate(data_loader_val):
            #if idx not in indices_to_plot:
            #    continue
            sample = sample.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            gt_box_coords = targets[0]['boxes']
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            tic = time.time()
            outputs = model(sample)
            runtime += (time.time() - tic)
            results = postprocess['bbox'](outputs, target_sizes)[0]

            # Write results in WIDER format
            relpath = data_loader_val.dataset.img_paths[idx].split("images/")[1]
            event, img_name = relpath.split("/")
            print(event)
            print(img_name)
            img_name = img_name.replace(".jpg", "")
            # create dir if does not exist
            dir_name = join(results_dir, event)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            results_file = join(dir_name, img_name+".txt")
            write_results(results_file, img_name, results, args.box_detection_score)

            # Plot
            if args.plot_eval:
                box_coords = results["boxes"].cpu().numpy().astype(np.int)
                box_scores = results["scores"].cpu().numpy()
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(gt_box_coords.device)
                gt_box_coords = (gt_box_coords * scale_fct)
                gt_box_coords = box_ops.box_cxcywh_to_xyxy(gt_box_coords).cpu().numpy().astype(np.int)
                confident_boxes = box_scores > args.box_detection_score
                selected_box_coords = box_coords[confident_boxes]
                img = cv2.imread(dataset_val.get_img_path(idx))
                plot_utils.plot_bboxes(img, gt_box_coords, selected_box_coords)
    f.close()
    print("Mean runtime per image: {} sec".format(runtime/len(data_loader_val)))
