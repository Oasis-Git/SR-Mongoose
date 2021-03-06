import os
import time

import torch
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from resnet import Generator, ResidualConvBlock
from Config import config
from dataset import ImageDataset
import torch.nn as nn


def train():
    device = torch.device("cuda", 3)
    # import data
    train_dataloader, valid_dataloader = load_dataset()

    # import model
    model = Generator()
    checkpoint = torch.load('../results/results_prefix/SRResNet_baseline/g-best.pth')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.cuda(device)

    be_trained = Fit()
    be_trained = be_trained.cuda(device)

    psnr_criterion, pixel_criterion = define_loss(device)
    optimizer = define_optimizer(be_trained)

    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "Logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    print("Start train SRResNet model.")
    for epoch in range(config.start_epoch, config.epochs):
        train_epoch(model, be_trained, train_dataloader, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer, device)

        psnr = valid_epoch(model, be_trained, valid_dataloader, psnr_criterion, epoch, writer, device)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save(be_trained.state_dict(), os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth"))
        if is_best:
            torch.save(be_trained.state_dict(), os.path.join(results_dir, "g-best.pth"))

    # Save the generator weight under the last Epoch in this stage
    torch.save(be_trained.state_dict(), os.path.join(results_dir, "g-last.pth"))
    print("End train Fit model.")



def train_epoch(outmodel, model, train_dataloader, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer, device):
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

        lr = lr.cuda(device)
        hr = hr.cuda(device)

        with torch.no_grad():
            hr = outmodel.downsample(hr)

        # print("train input ", lr)

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
        if index % config.print_frequency == 0 and index != 0:
            progress.display(index)


def valid_epoch(outmodel, model, valid_dataloader, psnr_criterion, epoch, writer, device):
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.cuda(device)
            hr = hr.cuda(device)

            with torch.no_grad():
                hr = outmodel.downsample(hr)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), hr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % config.print_frequency == 0:
                progress.display(index)

        writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
        # Print evaluation indicators.
        print(f"* PSNR: {psnres.avg:4.2f}.\n")

    return psnres.avg


def define_loss(device) -> [nn.MSELoss, nn.MSELoss]:
    psnr_criterion = nn.MSELoss().cuda(device)
    pixel_criterion = nn.MSELoss().cuda(device)

    return psnr_criterion, pixel_criterion


def define_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def load_dataset() -> [DataLoader, DataLoader]:
    train_datasets = ImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "train")
    valid_datasets = ImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "valid")
    print("Train dataset size is " + str(train_datasets.getsize()))
    print("Valid dataset size is " + str(valid_datasets.getsize()))
    # Make it into a data set type supported by PyTorch
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True)

    return train_dataloader, valid_dataloader


class Fit(nn.Module):
    def __init__(self) -> None:
        super(Fit, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.LeakyReLU()
        trunk = []
        for _ in range(12):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU()
        )

        self.conv_down2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU()
        )

        self.conv_down3 = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU()
        )

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        add = self.trunk(res)
        res = res + add
        res = self.conv_down1(res)
        res = self.conv_down2(res)
        res = self.conv_down3(res)

        return res

class Direct(nn.Module):
    def __init__(self) -> None:
        super(Direct, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.LeakyReLU()
        trunk = []
        for _ in range(12):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU()
        )

        self.conv_down2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU()
        )

        self.conv_down3 = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU()
        )

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        add = self.trunk(res)
        res = res + add
        res = self.conv_down1(res)
        res = self.conv_down2(res)
        res = self.conv_down3(res)

        return res

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
