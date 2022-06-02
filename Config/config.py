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
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn


"""
my_down_up is my model for hr to hr similar to aeted
use model: resnet.mygenerator
"""

"""
aeted is model work in 2018
use model: aetad.4k
"""

# Random seed to maintain reproducible results
torch.manual_seed(0)
# Use GPU for training by default
device = torch.device("cuda", 2)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train_srresnet_up"
# Experiment name, easy to save weights and log files
exp_name = "SRResNet_mine_down_up"
sample_ratio = 0.5
const_ratio = 1.0

if mode == "train_srresnet_down":
    # Dataset address

    train_image_dir = "/home/ayw/Documents/DIV2K_train_HR"
    valid_image_dir = "/home/ayw/Documents/DIV2K_valid_HR"

    image_size = 96
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = False
    strict = True
    start_epoch = 0
    resume_weight = ""

    # Total num epochs
    epochs = 3000

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # Print the training log every one hundred iterations
    print_frequency = 1000

if mode == "train_srresnet_up":
    # Dataset address
    train_image_dir = "/home/ayw19/data/DIV2K_train_HR"
    valid_image_dir = "/home/ayw19/data/DIV2K_valid_HR"


    image_size = 96
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = False
    strict = True
    start_epoch = 0
    resume_weight = ""

    # Total num epochs
    epochs = 3000

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # Print the training log every one hundred iterations
    print_frequency = 1000

if mode == "train_srgan":
    # Dataset address
    train_image_dir = "data/ImageNet/SRGAN/train"
    valid_image_dir = "data/ImageNet/SRGAN/valid"

    image_size = 96
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = True
    strict = False
    start_epoch = 0
    resume_d_weight = ""
    resume_g_weight = "results/SRResNet_baseline/g-last.pth"

    # Total num epochs
    epochs = 9

    # Loss function weight
    pixel_weight = 1.0
    content_weight = 1.0
    adversarial_weight = 0.001

    # Adam optimizer parameter
    d_model_lr = 1e-4
    g_model_lr = 1e-4
    d_model_betas = (0.9, 0.999)
    g_model_betas = (0.9, 0.999)

    # MultiStepLR scheduler parameter
    d_optimizer_step_size = epochs // 2
    g_optimizer_step_size = epochs // 2
    d_optimizer_gamma = 0.1
    g_optimizer_gamma = 0.1

    # Print the training log every one hundred iterations
    print_frequency = 1000

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/g-best.pth"

if mode == "valid2":
    # Test data address
    lr_image_path = f"data/Camelyon/test/Dakewe_slide_20210923_8/LR/00000288.bmp"
    sr_image_path = f"00000288.bmp"
    hr_image_path = f"data/Camelyon/test/Dakewe_slide_20210923_8/HR/00000288.bmp"

    model_dir = f"C:/Code/Real-ESRGAN/experiments/finetune_train_RealESRNet_x2_Camelyon_single_scale_300K_ImageSize256_BS8_LR0.0002/models"


if mode == "valid3":
    # Test data address
    lr_dir = f"data/Camelyon/test/Dakewe_slide_20210923_8/LR"
    sr_dir = f"data/Camelyon/test/Dakewe_slide_20210923_8/SRx2"
    hr_dir = f"data/Camelyon/test/Dakewe_slide_20210923_8/HR"

    model_path = f"C:/Code/Real-ESRGAN/experiments/train_RealESRGAN_x2_Camelyon_single_scale_400K_ImageSize128_BS16/net_g_latest.pth"


if mode == "valid4":
    # Test data address
    lr_image_path = f"C:/Code/Image-Quality-Assesment/Python/libsvm/python/test/Dakewe_slide_20210923_8/LR/00000288.bmp"
    sr_image_path = f"C:/Code/Image-Quality-Assesment/Python/libsvm/python/test/Dakewe_slide_20210923_8/SRx2/00000288.bmp"

    model_path = f"C:/Code/Real-ESRGAN/experiments/train_PMIGAN_x2_Camelyon_single_scale_400K_ImageSize128_BS16_LR0.0001/models/net_g_latest.pth"


# Camelyon: camelyon17_slide_1_46288_28080_000.bmp
# Dakewe: 00000288.bmp
