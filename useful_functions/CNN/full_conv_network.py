"""
FCN (Fully Convolutional Network) on PASCAL VOC2011 -- corrected, runnable version.

Fixes applied relative to the original draft are called out inline with "FIX:" comments.
See docs/fcn_code_review.md for the full step-by-step explanation.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import VOCSegmentation
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# 1. Dataset download
# ---------------------------------------------------------------------------
# FIX: the original cell used Jupyter shell-magic (`!wget`, `!mkdir`, `!tar`),
# which only runs inside a notebook, not as a plain .py script. It also only
# checked for "VOCdevkit" (not the VOC2011 subfolder), so a partially-extracted
# archive from a different year could false-positive the "already prepared" check.
# `download_and_extract_archive` is pure Python, works in scripts and notebooks,
# and mirrors exactly what VOCSegmentation(download=True) does internally.

ROOT_DIR = "./VOCSegmentation/2011"
DEVKIT_PATH = os.path.join(ROOT_DIR, "VOCdevkit", "VOC2011")


def prepare_dataset():
    if os.path.exists(DEVKIT_PATH):
        print("The dataset is already prepared.")
        return
    print("Downloading and extracting the dataset...")
    download_and_extract_archive(
        url="http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        download_root=ROOT_DIR,
        filename="VOCtrainval_25-May-2011.tar",
    )
    print(f"The dataset is ready at {DEVKIT_PATH}")


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
# Image Shape: [3, 224, 224] (Channel, Height, Width)
# Target Shape: [224, 224]   (class index per pixel, 0-20 = object/background, 255 = void/ignore)
#
# FIX: the original TargetToTensor() did `target[target > 20] = 0`, which silently
# relabels every void/boundary pixel (255) as class 0 (background). That corrupts
# the ground truth: ambiguous edge pixels get treated as confident "background"
# labels, which both biases training and inflates/deflates class 0's IoU.
# The correct approach is to LEAVE the 255 pixels alone and exclude them from the
# loss via CrossEntropyLoss(ignore_index=255) -- see the model training section.
#
# FIX: the original one-hot-encoded the target to (21, 224, 224) float. This is
# unnecessary (nn.CrossEntropyLoss accepts plain class-index targets), wastes
# memory, and -- combined with the ignore_index requirement above -- doesn't
# compose cleanly with CrossEntropyLoss's ignore_index, which only works on
# integer class-index targets, not one-hot/probability targets. We keep the
# target as a plain (H, W) LongTensor of class indices instead.
#
# FIX: `transforms.Resize` defaults to bilinear interpolation. Applying that to
# a segmentation mask blends class indices at edges (e.g. averaging class 5 and
# class 12 can produce a nonsense value), silently corrupting labels. Masks must
# be resized with NEAREST interpolation so every output pixel is still a valid,
# untouched class index.

batch_size = 16
num_classes = 21

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def target_to_tensor(target):
    # target: PIL Image, mode "P" (palette), pixel value == class index directly
    target = np.array(target, dtype=np.int64)   # (224, 224) int64, values in {0..20, 255}
    target = torch.from_numpy(target).long()    # (224, 224) LongTensor -- kept as class indices, NOT one-hot
    return target


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),                       # bilinear is fine for RGB images
    transforms.ToTensor(),                                 # (3, 224, 224), float in [0, 1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # FIX: normalize to match the pretrained backbone
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),  # FIX: nearest, not bilinear
    transforms.Lambda(target_to_tensor),
])


def build_dataloaders():
    train_dataset = VOCSegmentation(
        root=ROOT_DIR, year="2011", image_set="train", download=False,
        transform=image_transform, target_transform=target_transform,
    )
    valid_dataset = VOCSegmentation(
        root=ROOT_DIR, year="2011", image_set="val", download=False,
        transform=image_transform, target_transform=target_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_dataloader, valid_dataloader


# ---------------------------------------------------------------------------
# 3. Model definition
# ---------------------------------------------------------------------------
# FIX: the original passed `backbone=resnet18` -- the constructor function
# itself, never instantiated. Even instantiated, a full resnet18 forward pass
# ends in avgpool + fc and returns (N, 1000) classification logits, not a
# (N, 512, 7, 7) feature map. We build a proper feature-extractor by dropping
# the last two children (avgpool, fc) and loading ImageNet-pretrained weights,
# since the FCNhead is designed around a 512-channel, stride-32 feature map.

def build_backbone():
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # net.children(): [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
    # drop the last two (avgpool, fc) to keep a spatial feature map
    backbone = nn.Sequential(*list(net.children())[:-2])
    return backbone


class FCN(nn.Module):
    def __init__(self, backbone, num_classes=21):
        super().__init__()
        self.backbone = backbone
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.FCNhead = nn.Sequential(
            # input: (N, 512, 7, 7)
            nn.Conv2d(512, 128, kernel_size=3, padding=1),   # (N, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1),      # (N, 21, 7, 7)
        )

    def forward(self, x):
        # x: (N, 3, 224, 224)
        features = self.backbone(x)          # (N, 512, 7, 7) -- resnet18 has output stride 32 (224 / 32 = 7)
        scores = self.FCNhead(features)       # (N, 21, 7, 7)  -- coarse per-class score map
        out = F.interpolate(
            scores, size=x.shape[-2:], mode="bilinear", align_corners=False
        )                                       # (N, 21, 224, 224) -- upsampled back to input resolution
        return out


# ---------------------------------------------------------------------------
# 4. Metric: mean IoU
# ---------------------------------------------------------------------------
# FIX: the original allocated `self.confusion_matrix` but never wrote into it --
# `label_accuracy_score` built a fresh local `hist` every call and returned
# per-batch scores. Averaging per-batch mIoU over many small batches is NOT the
# same as computing mIoU from one confusion matrix accumulated over an entire
# epoch (rare classes that are absent from some batches produce NaNs that skew
# the nanmean). We fix this with an explicit update()/get_scores()/reset() cycle:
# accumulate every batch's confusion matrix during the epoch, then read the
# score once at the end.
#
# Note this also naturally handles the void/ignore label: `_fast_hist`'s mask
# `(labels_true >= 0) & (labels_true < n_class)` excludes any pixel with value
# 255, since 255 is not < num_classes(=21). No extra ignore-handling needed here.

class mIoU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def _fast_hist(self, labels_true, labels_pred, n_class):
        mask = (labels_true >= 0) & (labels_true < n_class)
        hist = np.bincount(
            n_class * labels_true[mask].astype(int) + labels_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, labels_true, labels_pred):
        """labels_true, labels_pred: numpy arrays of any matching shape (e.g. (N, H, W))."""
        labels_true = labels_true.flatten()
        labels_pred = labels_pred.flatten()
        self.confusion_matrix += self._fast_hist(labels_true, labels_pred, self.num_classes)

    def get_scores(self):
        """Computes scores from the ACCUMULATED confusion matrix:
            - overall accuracy
            - mean class accuracy
            - mean IoU
            - frequency-weighted average accuracy
        """
        hist = self.confusion_matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.diag(hist).sum() / hist.sum()
            acc_cls = np.diag(hist) / hist.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            mean_iu = np.nanmean(iu)
            freq = hist.sum(axis=1) / hist.sum()
            fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


# ---------------------------------------------------------------------------
# 5. Trainer
# ---------------------------------------------------------------------------
# FIX summary for this section (see docs/fcn_code_review.md for full detail):
#   - `self.optmizer` was a typo AND was called like a function (`self.optmizer()`,
#     `self.optmizer().step()`) instead of using its methods
#     (`self.optimizer.zero_grad()` / `self.optimizer.step()`), and those calls
#     were in the wrong order relative to `loss.backward()`.
#   - `self.model(images).forward()` called `.forward()` on a Tensor, which
#     doesn't exist -- `self.model(images)` already runs the forward pass.
#   - `self.model.test()` isn't a real nn.Module method -- it's `self.model.eval()`,
#     and validation should run inside `torch.no_grad()`.
#   - the trainer rebuilt a second, incorrectly-constructed FCN(resnet18, ...)
#     instead of reusing a properly built model, and never moved it to device.
#   - CrossEntropyLoss() had no `ignore_index=255`, so void pixels would still
#     leak into the loss even after the label bug above is fixed.
#   - metrics were computed on raw logits/one-hot targets without `argmax` or
#     `.cpu().numpy()`, and the epoch loop hardcoded `range(100)` instead of
#     using `self.epochs`.
#   - per-batch plotting produced an extremely noisy chart; we now aggregate
#     one point per epoch.

class SegmentationTrainer:
    def __init__(self, model, train_loader, valid_loader, num_classes=21, lr=1e-4, epochs=10, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)  # FIX: ignore void/boundary pixels
        self.metrics = mIoU(num_classes=num_classes)
        self.data_loader = {"train": train_loader, "valid": valid_loader}
        self.epochs = epochs

    def run_epoch(self, phase):
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()
        loader = self.data_loader[phase]
        self.metrics.reset()
        running_loss = 0.0

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            for images, targets in loader:
                images = images.to(self.device)   # (N, 3, 224, 224)
                targets = targets.to(self.device)  # (N, 224, 224) long, values in {0..20, 255}

                logits = self.model(images)         # (N, 21, 224, 224)
                loss = self.criterion(logits, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * images.size(0)

                preds = logits.argmax(dim=1)         # (N, 224, 224) predicted class index per pixel
                self.metrics.update(targets.cpu().numpy(), preds.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        acc, _acc_cls, mean_iu, _fwavacc = self.metrics.get_scores()
        return epoch_loss, acc, mean_iu

    def training(self):
        history = {
            "train_loss": [], "train_acc": [], "train_miou": [],
            "valid_loss": [], "valid_acc": [], "valid_miou": [],
        }
        for _epoch in tqdm(range(self.epochs)):
            tr_loss, tr_acc, tr_miou = self.run_epoch("train")
            va_loss, va_acc, va_miou = self.run_epoch("valid")
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["train_miou"].append(tr_miou)
            history["valid_loss"].append(va_loss)
            history["valid_acc"].append(va_acc)
            history["valid_miou"].append(va_miou)
        self.visualize_result(history)
        return history

    def visualize_result(self, history):
        _fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_acc"], label="train_accuracy")
        axes[0].plot(history["valid_acc"], label="valid_accuracy")
        axes[0].set_title("Pixel Accuracy")
        axes[0].set_xlabel("epoch")
        axes[0].legend()
        axes[1].plot(history["train_miou"], label="train_miou")
        axes[1].plot(history["valid_miou"], label="valid_miou")
        axes[1].set_title("Mean IoU")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prepare_dataset()
    train_dataloader, valid_dataloader = build_dataloaders()

    model = FCN(backbone=build_backbone(), num_classes=num_classes)

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        num_classes=num_classes,
        lr=1e-4,
        epochs=10,  # NOTE: original draft used 100; lowered for a tractable first run, raise as needed
    )
    trainer.training()
