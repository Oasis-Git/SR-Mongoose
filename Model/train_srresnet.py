# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""File description: Initialize the SRResNet model."""
import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import *
from .dataset import ImageDataset
from .model import Generator


def main():
    print("Load train dataset and valid dataset...")
    train_dataloader, valid_dataloader = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    print("Build SRResNet model...")
    model = build_model()
    print("Build SRResNet model successfully.")

    print("Define all loss functions...")
    psnr_criterion, pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    print("Define all optimizer functions...")
    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    print("Check whether the training weight is restored...")
    resume_checkpoint(model)
    print("Check whether the training weight is restored successfully.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", exp_name)
    results_dir = os.path.join("results", exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    print("Start train SRResNet model.")
    for epoch in range(start_epoch, epochs):
        train(model, train_dataloader, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)

        psnr = validate(model, valid_dataloader, psnr_criterion, epoch, writer)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save(model.state_dict(), os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(results_dir, "g-best.pth"))

    # Save the generator weight under the last Epoch in this stage
    torch.save(model.state_dict(), os.path.join(results_dir, "g-last.pth"))
    print("End train SRResNet model.")


def load_dataset() -> [DataLoader, DataLoader]:
    train_datasets = ImageDataset(train_image_dir, image_size, upscale_factor, "train")
    valid_datasets = ImageDataset(valid_image_dir, image_size, upscale_factor, "valid")
    # Make it into a data set type supported by PyTorch
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return train_dataloader, valid_dataloader


def build_model() -> nn.Module:
    model = Generator().to(device)

    return model


def define_loss() -> [nn.MSELoss, nn.MSELoss]:
    psnr_criterion = nn.MSELoss().to(device)
    pixel_criterion = nn.MSELoss().to(device)

    return psnr_criterion, pixel_criterion


def define_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), model_lr, model_betas)

    return optimizer


def resume_checkpoint(model) -> None:
    if resume:
        if resume_weight != "":
            # Get pretrained model state dict
            pretrained_state_dict = torch.load(resume_weight)
            model_state_dict = model.state_dict()
            # Extract the fitted model weights
            new_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict.items()}
            # Overwrite the pretrained model weights to the current model
            model_state_dict.update(new_state_dict)
            model.load_state_dict(model_state_dict, strict=strict)


def train(model, train_dataloader, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_dataloader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in train mode.
    model.train()

    end = time.time()
    for index, (lr, hr) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Initialize the generator gradient
        model.zero_grad()

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Gradient zoom
        scaler.scale(loss).backward()
        # Update generator weight
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        losses.update(loss.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Writer Loss to file
        writer.add_scalar("Train/Loss", loss.item(), index + epoch * batches + 1)
        if index % print_frequency == 0 and index != 0:
            progress.display(index)


def validate(model, valid_dataloader, psnr_criterion, epoch, writer) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), hr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % print_frequency == 0:
                progress.display(index)

        writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
        # Print evaluation indicators.
        print(f"* PSNR: {psnres.avg:4.2f}.\n")

    return psnres.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()