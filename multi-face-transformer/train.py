"""
Train functions used in main.py
"""
import math
import sys
from typing import Iterable
import datasets.transforms as T
from datasets.transforms import normalize_transform
import datetime
import torch
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets.wider.WIDERFaceDataset import WIDERFaceDataset
from os.path import join
import time

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # Code taken from: https://github.com/facebookresearch/detrCopyright

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args, model, criterion):
    device = torch.device(args.device)

    # Set model, optimizer and scheduler
    model.to(device)
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Set dataset and data loader
    if args.freeze_encoder:
        max_size = 1333
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    else:
        max_size = 608
        scales = [480, 512, 544, 576, 608]

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=max_size),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=max_size),
            ])
        ),
        normalize_transform,
    ])

    dataset_file = join(join(args.dataset_path, "annotations"), "WIDER_train_annotations.txt")
    dataset_train = WIDERFaceDataset(args.dataset_path, dataset_file, 'train',
                                     img_transforms=transform)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    params_to_freeze = []
    if args.freeze_backbone:
        params_to_freeze.append("backbone")
    if args.freeze_encoder:
        params_to_freeze.append("transformer.encoder")
        params_to_freeze.append("input_proj")
    if args.freeze_decoder:
        params_to_freeze.append("transformer.decoder")

    for name, parameter in model.named_parameters():
        for phrase in params_to_freeze:
            if phrase in name:
                parameter.requires_grad_(False)
                print("Freezing param: [{}]".format(name))

    # Train
    print("Start training")
    save_every = args.epochs // 6
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        print(train_stats)

        if args.output_dir:
            checkpoint_paths = [output_dir / '{}_checkpoint_final.pth'.format(args.checkpoint_prefix)]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % save_every == 0:
                checkpoint_paths.append(
                    output_dir / '{}_epoch{}_checkpoint.pth'.format(args.checkpoint_prefix, epoch + 1))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




