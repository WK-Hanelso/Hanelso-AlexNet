import os
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Training Configs ====================================================

epochs = 60
batch_size = 64
learning_rate = 5e-4
train_data_dir = "DATA/PokemonData"
val_data_dir = "DATA/PokemonData"
test_data_dir = "DATA/PokemonData"
save_dir = "results"
val_split_percent = 0.2

shuffle = True  # DataLoader.shuffle
num_workers = min( 12, os.cpu_count()) # DataLoader.num_workers
n_classes = 149
loss_label_smoothing = 0.1
drop_out_ratio = 0.0

# ImageNet 통계(그대로 유지)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# 학습용 변환: 약한 증강 + 224 크롭
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(3/4, 4/3), interpolation=InterpolationMode.BICUBIC),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
# ])

transform_train = transforms.Compose([
  transforms.Resize((224, 224)),  # Resize to a common size (adjust as needed)
  transforms.RandomHorizontalFlip(0.5),
  transforms.RandomVerticalFlip(p=0.5),
  transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
  transforms.ToTensor(),
  transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)  # Normalize for ImageNet models
])

transform_eval= transforms.Compose([
  transforms.Resize((224, 224)),  # Resize to a common size (adjust as needed)
  transforms.ToTensor(),
  transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)  # Normalize for ImageNet models
])

# 검증/테스트용 변환: 크기만 맞추고 중앙 크롭
# transform_eval = transforms.Compose([
#     transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
# ])

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
# device = torch.device( "cpu" )

# ======================================================================
