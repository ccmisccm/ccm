# Define a simple CNN classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cfg
import datetime
import os
import numpy as np



classes = ['Non-Ectopic Beats', 'Superventrical Ectopic', 'Ventricular Beats', 'Unknown', 'Fusion Beats']
classes_idx = ['1','2','3','4','5','6','7']
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

class ECG_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 6)
        self.conv2 = nn.Conv1d(64, 64, 6)
        self.conv3 = nn.Conv1d(64, 64, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48832, 100)
        self.fc2 = nn.Linear(100, 7)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # 添加 forward_without_fc 用于提取中间特征
    def forward_without_fc(self, x):
        x = x.view(x.size(0), 1, -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        # 返回 fc1 之前的特征
        return x




def eval(model, real_test_loader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(real_test_loader):
            # get the inputs; data is a list of [inputs, labels]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs, labels = data
            inputs = inputs.to(device).double()  # 将输入移动到 device
            labels = labels.to(device).long()  # 同时将标签移动到 device

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_test_loss = total_loss / len(real_test_loader)
        epoch_test_acc = correct / total

    print('=====================================================')
    print(f'Epoch {epoch + 1}, test loss = {epoch_test_loss}, test acc = {epoch_test_acc}')
    print('=====================================================')


def eval_single_class(model, real_test_loader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(real_test_loader):
            # get the inputs; data is a list of [inputs, labels]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs, labels = data
            inputs = inputs.to(device).double()  # 将输入移动到 device
            labels = labels.to(device).long()  # 同时将标签移动到 device

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))


def final_eval(model, real_test_loader, criterion, filename="test_cm"):
    nb_classes = 7
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    y_preds = []
    y_trues = []

    # 预测过程：遍历测试数据，收集预测结果和真实标签
    with torch.no_grad():
        for i, data in enumerate(real_test_loader):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs, labels = data
            inputs = inputs.to(device).double()  # 将输入移动到 device
            labels = labels.to(device).long()  # 同时将标签移动到 device

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            # Append batch prediction results
            predlist = torch.cat([predlist, predictions.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

            y_preds.append(predictions)
            y_trues.append(labels)

    # 计算混淆矩阵
    cm = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print("Confusion Matrix:\n", cm)
    cm_df = pd.DataFrame(cm,
                         index=classes_idx,  # 例如：['1','2','3','4','5','6','7']
                         columns=classes_idx)

    # 绘制混淆矩阵热力图
    fig = plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='cubehelix_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  # 防止标签被截断

    # 先使用 fig.savefig 保存热力图，避免 plt.show() 清空图像
    heatmap_pdf = f'{filename}.pdf'
    fig.savefig(heatmap_pdf)
    print(f"Heatmap saved to {heatmap_pdf}")
    # plt.show()  # 可选：显示图像

    # 计算每个类别的准确率
    class_accuracy = 100 * cm.diagonal() / cm.sum(1)
    print("Per-class Accuracy (%):", class_accuracy)

    # Flatten predictions and labels，用于生成分类报告
    y_preds_flatten = [label for sublist in y_preds for label in sublist]
    y_trues_flatten = [label for sublist in y_trues for label in sublist]

    # 定义一个新的类别名称列表，长度为 7，与实际类别数一致
    target_names_7 = [
        "Non-Ectopic Beats",         # label 0 (原始标签 0)
        "Superventricular Ectopic",  # label 1 (原始标签 5)
        "Ventricular Beats",         # label 2 (原始标签 6)
        "Unknown",                   # label 3 (原始标签 7)
        "Fusion Beats",              # label 4 (原始标签 8)
        "Extra Class 6",             # label 5 (原始标签 9)
        "Extra Class 7"              # label 6 (原始标签 10)
    ]
    y_trues_flatten_tensor = torch.tensor(y_trues_flatten)
    y_preds_flatten_tensor = torch.tensor(y_preds_flatten)
    report = classification_report(
        y_trues_flatten_tensor.cpu().numpy(), y_preds_flatten_tensor.cpu().numpy(),
        labels=list(range(7)),
        target_names=target_names_7
    )

    print(report)

    # -------------------------------
    # 保存评估结果到带有时间戳的文件中
    # -------------------------------
    import datetime
    import os

    # 获取当前时间戳
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    # 解析参数，并确保 exp_name 存在
    args = cfg.parse_args()
    assert args.exp_name

    # 在 args.exp_name 目录下创建一个名为 "class_result" 的子文件夹
    result_dir = os.path.join(args.exp_name, "class_result")
    os.makedirs(result_dir, exist_ok=True)

    # 构造评估结果文本文件的完整路径
    result_file = os.path.join(result_dir, f"{filename}_{timestamp}.txt")
    with open(result_file, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\n")
        f.write("Per-class Accuracy (%):\n")
        f.write(np.array2string(class_accuracy))
        f.write("\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved evaluation results to {result_file}")

    # 将混淆矩阵热力图保存到同一目录下
    heatmap_file = os.path.join(result_dir, f"{filename}_{timestamp}_heatmap.pdf")
    fig.savefig(heatmap_file)
    print(f"Saved confusion matrix heatmap to {heatmap_file}")


