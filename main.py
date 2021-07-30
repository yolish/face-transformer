"""
Entry point for training and testing FaceTransformer
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils, box_ops












import time
from models.face_transformer import FaceTransformer, FaceAttrCriterion, postprocess
from os.path import join
import math
import sys
from torchvision import transforms
import pandas as pd
from torchvision.datasets import CelebA



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train, eval or test")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("config_file", help="path to configuration file")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} FaceTransformer".format(args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = FaceTransformer(config, args.backbone_path).to(device)
    # Load the checkpoin
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))
    else:
        assert args.mode == 'train'

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set loss
        criterion = FaceAttrCriterion(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        transform = transforms.Compose([
                                        transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

        dataset = CelebA(args.dataset_path, split=args.mode,
                         target_type=["attr", "bbox", "landmarks", "identity"],
                         transform=transform,
                         target_transform=None, download=False)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        max_norm = config.get("max_norm")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, (samples, targets) in enumerate(dataloader):

                attributes = targets[0].transpose(0,1)
                bbox = targets[1]
                landmarks = targets[2]
                identity = targets[3]
                targets_dict = {}
                for i, attr_name in enumerate(dataset.attr_names):
                    targets_dict[attr_name] = attributes[i, :]\
                        .unsqueeze(1).to(device).to(dtype=torch.float32)

                targets_dict["identity"] = identity.to(device).to(dtype=torch.int64)

                # transform to relative [0, 1] coordinates
                bbox = bbox

                img_h, img_w = samples.shape[2:]
                scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(samples.shape[0], 1)
                bbox = bbox / scale_fct
                targets_dict["bbox"] = bbox.to(device)

                targets_dict["landmarks"] = landmarks.to(device)
                samples = samples.to(device)

                optimizer.zero_grad()
                outputs = model(samples)
                loss, loss_dict = criterion(outputs, targets_dict)

                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    print(loss)
                    sys.exit(1)

                loss.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

                # Collect for recoding and plotting
                running_loss += loss.item()
                loss_vals.append(loss.item())
                sample_count.append(n_total_samples)

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    logging.info("[Batch-{}/Epoch-{}] loss dict: {}".format(batch_idx+1, epoch+1, loss.item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

    else: # Test or Eval
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        dataset = CelebA(args.dataset_path, split=args.mode,
                         target_type=["attr", "bbox", "landmarks", "identity"],
                         transform=transform,
                         target_transform=None, download=False)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = {}
        properties = config.get("properties")

        res = {"img":[], "time":[]}
        for attr in properties:
            res[attr] = []

        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(dataloader):
                samples = samples.to(device)
                tic = time.time()
                outputs = model(samples)
                toc = time.time()
                res["time"].append(toc-tic)
                res["img"].append(dataloader.dataset.paths[batch_idx])
                proc_outputs = postprocess(outputs, config)
                for property, val in proc_outputs:
                    res[property].append(val)
                    #TODO add real value

        df = pd.Dataframe(res)
        out_file = args.chekpoint_path + "_predictions.csv"
        df.to_csv(out_file)
        print("Predictions written to: {}".format(out_file))







