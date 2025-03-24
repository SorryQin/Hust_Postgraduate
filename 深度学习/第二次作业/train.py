import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models.pspnet import PSPNet
from models.deeplabv3 import DeepLabv3
from models.ccnet import CCNet
from dataloader import get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()


def evaluate_miou(model, val_loader, device,num_classes):
    model.eval()
    total_miou = 0
    total_loss = 0
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # 计算损失
            loss_ce = criterion_ce(outputs, masks)
            # loss_dice = criterion_dice(outputs, masks)
            # loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = loss_ce
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            preds_np = preds.detach().cpu().numpy()
            masks_np = masks.cpu().numpy()

            # 计算mIoU（简化版）
            iou_per_class = []
            for cls in range(num_classes):
                tp = (( preds_np== cls) & (masks_np == cls)).sum()
                fp = (( preds_np == cls) & (masks_np != cls)).sum()
                fn = (( preds_np != cls) & (masks_np == cls)).sum()
                iou = tp / (tp + fp + fn + 1e-5)
                iou_per_class.append(iou)
            total_miou += np.nanmean(iou_per_class)

            # confusion_matrix = get_confusion_matrix_for_3d(masks_np, preds_np, class_num=6)
            # pos = confusion_matrix.sum(1)
            # res = confusion_matrix.sum(0)
            # tp = np.diag(confusion_matrix)
            # IU_array = (tp / np.maximum(1.0, pos + res - tp))
            # mean_IU = IU_array.mean()
            # total_miou += mean_IU
            # iou = calculate_iou(preds, masks)
            # total_iou += iou
    return total_miou / len(val_loader), total_loss/len(val_loader)


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
            Calcute the confusion matrix by given label and pred
            :param gt_label: the ground truth label
            :param pred_label: the pred label
            :param class_num: the number of class
            :return: the confusion matrix
            """
    index = (gt_label * class_num + pred_label).astype('int32')

    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def get_confusion_matrix_for_3d(gt_label, pred_label, class_num):
    confusion_matrix = np.zeros((class_num, class_num))

    for sub_gt_label, sub_pred_label in zip(gt_label, pred_label):
        sub_gt_label = sub_gt_label[sub_gt_label != 255]
        sub_pred_label = sub_pred_label[sub_pred_label != 255]
        cm = get_confusion_matrix(sub_gt_label, sub_pred_label, class_num)
        confusion_matrix += cm
    return confusion_matrix


def train_model(model_name, num_classes, device='cuda', epochs=50, aug_mode='base'):
    # 初始化模型
    if model_name == 'PSPNet':
        model = PSPNet(num_classes).to(device)
    elif model_name == 'DeepLabv3':
        model = DeepLabv3(num_classes).to(device)
    elif model_name == 'CCNet':
        model = CCNet(num_classes).to(device)

    # 数据加载
    train_loader = get_dataloader('data',img_size=128,phase='train', aug_mode=aug_mode)
    val_loader = get_dataloader('data',img_size=128,phase='val', aug_mode='none')

    # 优化器和损失
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()

    best_miou = 0
    no_improve = 0
    train_losses = []
    train_mious = []
    val_losses = []
    val_mious = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_total_miou = 0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_ce = criterion_ce(outputs, masks)
            # loss_dice = criterion_dice(outputs, masks)
            # loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = loss_ce
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            preds_np = preds.detach().cpu().numpy()
            masks_np = masks.cpu().numpy()

            # 计算mIoU
            iou_per_class = []
            for cls in range(num_classes):
                tp = (( preds_np== cls) & (masks_np == cls)).sum()
                fp = (( preds_np == cls) & (masks_np != cls)).sum()
                fn = (( preds_np != cls) & (masks_np == cls)).sum()
                iou = tp / (tp + fp + fn + 1e-5)
                iou_per_class.append(iou)
            train_total_miou += np.nanmean(iou_per_class)

            # confusion_matrix = get_confusion_matrix_for_3d(masks_np,  preds_np, class_num=6)
            # pos = confusion_matrix.sum(1)
            # res = confusion_matrix.sum(0)
            # tp = np.diag(confusion_matrix)
            # IU_array = (tp / np.maximum(1.0, pos + res - tp))
            # mean_IU = IU_array.mean()
            # train_total_miou += mean_IU

        scheduler.step()

        # 验证集评估
        avg_val_miou , avg_val_loss = evaluate_miou(model, val_loader, device,num_classes)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_miou = train_total_miou / len(train_loader)
        train_losses.append(avg_train_loss)
        train_mious.append(avg_train_miou)
        val_losses.append(avg_val_loss)
        val_mious.append(avg_val_miou)

        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}, Val mIoU: {avg_val_miou:.4f}')
        # 打印日志
        print(f"{model_name}_{aug_mode}:Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train mIoU: {avg_train_miou:.4f} | Val mIoU: {avg_val_miou:.4f}")

        # 早停机制
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            no_improve = 0
            torch.save(model.state_dict(), f'best_{model_name}_{aug_mode}.pth')
        else:
            no_improve += 1
            if no_improve >= 20:
                print("Early stopping!")
                break

        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # mIoU曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_mious, label='Train mIoU')
        plt.plot(val_mious, label='Val mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Training and Validation mIoU')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'training_curves_{model_name}_{aug_mode}.png')
        plt.close()


if __name__ == '__main__':
    # train_model('PSPNet', num_classes=6, epochs=100)
    train_model('CCNet', num_classes=6, epochs=100)
    # train_model('DeepLabv3', num_classes=6, epochs=100)
    # train_model('PSPNet', num_classes=6, epochs=100, aug_mode='full')
    # train_model('CCNet', num_classes=6, epochs=100, aug_mode='full')
    # train_model('DeepLabv3', num_classes=6, epochs=100,aug_mode='full')
    # train_model('PSPNet', num_classes=6, epochs=100, aug_mode='none')
    # train_model('CCNet', num_classes=6, epochs=100, aug_mode='none')
    # train_model('DeepLabv3', num_classes=6, epochs=100,aug_mode='none')
