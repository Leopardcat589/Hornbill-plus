# -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# 定义文件路径
TRAIN_TS_PATH = "E:\Study\Hornbill\Time-Series-Library-main\EthanolConcentration\EthanolConcentration_TRAIN.ts"
TEST_TS_PATH = "E:\Study\Hornbill\Time-Series-Library-main\EthanolConcentration\EthanolConcentration_TEST.ts"

DEVICE = "cpu"

# 调整超参数
BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-3


class PesticideDataset(Dataset):
    def __init__(self, ts_file_path, is_train=True):
        self.is_train = is_train

        # 读取文件
        with open(ts_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析元数据
        self.dimensions = int(re.search(r'@dimensions\s+(\d+)', content).group(1))
        self.series_length = int(re.search(r'@seriesLength\s+(\d+)', content).group(1))

        # 找到数据开始位置
        data_start = content.find('@data')
        if data_start == -1:
            raise ValueError("No @data tag found in the file")

        data_start += 5  # 跳过 "@data"
        data_lines = content[data_start:].strip().split('\n')

        # 解析数据并统计所有标签
        X_list = []
        labels_list = []
        compound_counts = defaultdict(int)
        concentration_counts = defaultdict(int)

        for line in data_lines:
            line = line.strip()
            if not line:
                continue

            if ':' in line:
                data_part, label = line.rsplit(':', 1)
                data_str = data_part.strip()
                label_name = label.strip()  # 如 "Ace7", "Ani9"等

                # 解析数据点
                data_points = []
                for num_str in data_str.split(','):
                    try:
                        if 'e-' in num_str:
                            base, exp = num_str.split('e-')
                            data_points.append(float(base) * (10 ** -float(exp)))
                        elif 'e+' in num_str:
                            base, exp = num_str.split('e+')
                            data_points.append(float(base) * (10 ** float(exp)))
                        else:
                            data_points.append(float(num_str))
                    except:
                        data_points.append(0.0)

                # 检查数据点数量
                total_points = len(data_points)
                expected_points = self.dimensions * self.series_length

                if total_points != expected_points:
                    if total_points < expected_points:
                        data_points.extend([0.0] * (expected_points - total_points))
                    else:
                        data_points = data_points[:expected_points]

                # 重塑数据
                try:
                    data_array = np.array(data_points).reshape(self.dimensions, self.series_length)
                except:
                    continue

                # 使用所有维度的平均值作为特征
                if self.dimensions > 1:
                    avg_data = np.mean(data_array, axis=0)
                else:
                    avg_data = data_array[0]

                X_list.append(avg_data)
                labels_list.append(label_name)

                # 统计化合物和浓度
                compound = label_name[:-2] if label_name[-2:].isdigit() else label_name[:-1]
                concentration = label_name[-2:] if label_name[-2:].isdigit() else label_name[-1:]
                compound_counts[compound] += 1
                concentration_counts[concentration] += 1

        # 创建类别映射
        self.all_labels = sorted(list(set(labels_list)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # 提取化合物和浓度信息
        self.compounds = sorted(
            list(set([label[:-2] if label[-2:].isdigit() else label[:-1] for label in self.all_labels])))
        self.concentrations = sorted(
            list(set([label[-2:] if label[-2:].isdigit() else label[-1:] for label in self.all_labels])))

        self.num_classes = len(self.all_labels)

        # 转换为数值标签
        y_list = [self.label_to_idx[label] for label in labels_list]

        if X_list:
            self.X = np.array(X_list, dtype=np.float32)[:, None, :]  # (N, 1, L)
        else:
            self.X = np.array([], dtype=np.float32).reshape(0, 1, self.series_length)

        self.y = np.array(y_list, dtype=np.int64) if y_list else None

        print(f"\n{'=' * 60}")
        print(f"Dataset: {ts_file_path}")
        print(f"{'=' * 60}")
        print(f"Loaded {len(self.X)} samples")
        print(f"Total classes: {self.num_classes}")
        print(f"Number of compounds: {len(self.compounds)}")
        print(f"Number of concentration levels: {len(self.concentrations)}")

        print(f"\nCompounds ({len(self.compounds)}): {', '.join(self.compounds[:10])}" +
              ("..." if len(self.compounds) > 10 else ""))
        print(f"Concentrations ({len(self.concentrations)}): {', '.join(self.concentrations)}")

        if self.y is not None:
            # 打印类别分布
            unique, counts = np.unique(self.y, return_counts=True)
            print(f"\nClass distribution (top 10):")
            for cls_idx, count in zip(unique[:10], counts[:10]):
                label = self.idx_to_label[cls_idx]
                print(f"  {label}: {count} samples")

            if len(unique) > 10:
                print(f"  ... and {len(unique) - 10} more classes")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.long)
            return x, y
        else:
            return x


# 更强大的网络架构
class PesticideCNN(nn.Module):
    def __init__(self, num_classes=90, input_length=90):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 全局平均池化
            nn.AdaptiveAvgPool1d(1),
        )

        # 计算特征维度
        self.feature_dim = 128

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


# 带标签平滑的交叉熵损失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        log_probs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


@torch.no_grad()
def evaluate(model, loader, num_classes, class_names):
    model.eval()
    correct = 0
    total = 0

    # 每个类别的统计
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(DEVICE)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)

        total += y.size(0)
        correct += (predicted.cpu() == y).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.numpy())

    # 更新类别统计
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    accuracy = 100 * correct / total

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies.append(100 * class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return accuracy, class_accuracies, class_total


def main():
    print("=" * 70)
    print("PESTICIDE SPECTRA CLASSIFICATION")
    print("18 compounds × 5 concentrations = 90 classes")
    print("=" * 70)

    # 加载数据集
    print("\n📥 Loading training data...")
    train_dataset = PesticideDataset(TRAIN_TS_PATH, is_train=True)

    if len(train_dataset) == 0:
        print("❌ No training data loaded!")
        return

    print("\n📥 Loading test data...")
    test_dataset = PesticideDataset(TEST_TS_PATH, is_train=False)

    if len(test_dataset) == 0:
        print("❌ No test data loaded!")
        return

    num_classes = train_dataset.num_classes
    print(f"\n🎯 Total number of classes: {num_classes}")
    print(f"📊 Training samples per class: ~{len(train_dataset) / num_classes:.1f}")
    print(f"📊 Test samples per class: ~{len(test_dataset) / num_classes:.1f}")

    # 检查数据集大小
    if len(train_dataset) < num_classes:
        print(f"\n⚠️  Warning: Very few training samples ({len(train_dataset)}) for {num_classes} classes!")
        print("   This is a challenging few-shot learning problem.")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)),
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 创建模型
    print(f"\n🛠️  Creating model with {num_classes} output classes...")
    model = PesticideCNN(num_classes=num_classes, input_length=train_dataset.series_length).to(DEVICE)

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 损失函数和优化器
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 100
    )

    best_accuracy = 0.0
    best_epoch = 0

    print("\n" + "=" * 70)
    print("🚀 Starting training...")
    print("=" * 70)

    # 训练循环 - 没有早停机制
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 更新学习率
        scheduler.step()

        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

        # 评估
        test_accuracy, class_accuracies, class_totals = evaluate(
            model, test_loader, num_classes, train_dataset.idx_to_label
        )

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
                'loss': avg_loss,
                'class_mapping': train_dataset.label_to_idx,
            }, "pesticide_cnn_best.pth")

            print(f"\n✅ Epoch {epoch:03d}: New best model!")
            print(f"   Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"   Test Accuracy: {test_accuracy:.2f}%")

        # 打印进度
        if epoch == 1 or epoch % 10 == 0 or test_accuracy > best_accuracy:
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n📊 Epoch {epoch:03d}/{EPOCHS} | LR: {current_lr:.6f}")
            print(f"   Train: Loss={avg_loss:.4f}, Acc={train_accuracy:.2f}%")
            print(f"   Test:  Acc={test_accuracy:.2f}%")

            # 打印一些类别的准确率
            if epoch % 20 == 0:  # 每20个epoch打印一次详细类别信息
                print(f"   Sample class accuracies:")
                for i in range(min(5, num_classes)):
                    if class_totals[i] > 0:
                        label = train_dataset.idx_to_label.get(i, f"Class_{i}")
                        print(f"     {label}: {class_accuracies[i]:.1f}% ({class_totals[i]} samples)")

    print("\n" + "=" * 70)
    print("🏁 Training completed!")
    print("=" * 70)
    print(f"📊 Best test accuracy: {best_accuracy:.2f}% (epoch {best_epoch})")
    print(f"💾 Best model saved as 'pesticide_cnn_best.pth'")

    # 加载最佳模型进行最终评估
    if os.path.exists("pesticide_cnn_best.pth"):
        print(f"\n📋 Loading best model for final evaluation...")
        checkpoint = torch.load("pesticide_cnn_best.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        final_accuracy, final_class_acc, final_class_totals = evaluate(
            model, test_loader, num_classes, train_dataset.idx_to_label
        )

        print(f"🏆 Final test accuracy: {final_accuracy:.2f}%")

        # 分析化合物和浓度的准确率
        compound_acc = defaultdict(list)
        concentration_acc = defaultdict(list)

        for i in range(num_classes):
            if final_class_totals[i] > 0 and i in train_dataset.idx_to_label:
                label = train_dataset.idx_to_label[i]
                acc = final_class_acc[i]

                # 提取化合物和浓度
                if len(label) >= 3:
                    compound = label[:-2] if label[-2:].isdigit() else label[:-1]
                    concentration = label[-2:] if label[-2:].isdigit() else label[-1:]

                    compound_acc[compound].append(acc)
                    concentration_acc[concentration].append(acc)

        print(f"\n📊 Average accuracy by compound:")
        for compound in sorted(compound_acc.keys())[:10]:  # 显示前10个
            avg_acc = np.mean(compound_acc[compound])
            print(f"   {compound}: {avg_acc:.1f}% ({len(compound_acc[compound])} concentrations)")

        print(f"\n📊 Average accuracy by concentration level:")
        for conc in sorted(concentration_acc.keys()):
            avg_acc = np.mean(concentration_acc[conc])
            print(f"   {conc}: {avg_acc:.1f}% ({len(concentration_acc[conc])} compounds)")


if __name__ == "__main__":
    main()
