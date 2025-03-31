
from DataLoader_2D_uav import mitbih_train, mitbih_test
from make_synDataLoader_2D_uav  import mixed_mitbih,syn_mitbih
from classification_model import *
from torch.utils import data
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')  # 或者使用其他可用的后端，如 'Agg', 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



def plot_tsne(model, data_loader, device='cpu', title='t-SNE Visualization', filename="test_cm"):
    """
    提取模型的特征，并使用 t-SNE 将高维特征降至二维后绘制散点图，并保存图像。
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, lbl in data_loader:
            inputs = inputs.to(device).double()
            # 使用模型的 forward_without_fc 提取特征
            feats = model.forward_without_fc(inputs)
            features.append(feats.cpu())
            labels.append(lbl.cpu())

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0).squeeze()  # 确保 labels 是一维

    # 使用 t-SNE 降维到二维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.numpy())

    # 绘制散点图，使用 tab10 以获得最多 10 个类别的清晰颜色
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels.numpy(),
        cmap='tab10',          # 这里使用 tab10
        alpha=0.7,
        edgecolor='none'
    )

    # 为 colorbar 指定 7 个刻度，对应 7 个类别
    cbar = plt.colorbar(scatter, ticks=range(7))
    cbar.set_label('Class Label')

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    # plt.show()  # 如果你想要立即显示图像，可取消注释

    # -------------------------------
    # 保存 t-SNE 图像到带有时间戳的文件中
    # -------------------------------
    import datetime
    import os

    # 获取当前时间戳
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    # 从配置中解析出 exp_name
    args = cfg.parse_args()
    assert args.exp_name, "请确保在命令行参数或配置文件中包含 exp_name"

    # 在 exp_name 目录下创建一个子文件夹，例如 "class_result"
    result_dir = os.path.join(args.exp_name, "class_result")
    os.makedirs(result_dir, exist_ok=True)

    # 构造保存 t-SNE 图的文件路径
    tsne_file = os.path.join(result_dir, f"{filename}_{timestamp}_tsne.png")
    plt.savefig(tsne_file, bbox_inches='tight', dpi=300)
    print(f"Saved t-SNE plot to {tsne_file}")




def get_data_loaders():
    # 加载训练集和测试集
    real_ecg = mitbih_train(n_samples=500, oneD=True)
    real_ecg_small = mitbih_train(n_samples=100, oneD=True)
    real_test_ecg = mitbih_test(oneD=True)
    mixed_ecg = mixed_mitbih(real_samples=200, syn_samples=300)
    syn_ecg = syn_mitbih(n_samples=500, reshape=True)

    # 使用 DataLoader 加载数据
    real_loader = data.DataLoader(real_ecg, batch_size=32, num_workers=4, shuffle=True)
    real_loader_small = data.DataLoader(real_ecg_small, batch_size=32, num_workers=4, shuffle=True)
    test_real_loader = data.DataLoader(real_test_ecg ,batch_size=32, num_workers=4, shuffle=True)
    mixed_loader = data.DataLoader(mixed_ecg, batch_size=32, num_workers=4, shuffle=True)
    syn_loader = data.DataLoader(syn_ecg, batch_size=32, num_workers=4, shuffle=True)
    return real_loader,real_loader_small, test_real_loader,syn_loader,mixed_loader




def train(model, train_data_loader, test_data_loader, epochs, criterion, optimizer, filename="test_cm"):
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)  # 将模型移动到 GPU

        for i, data in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            inputs = inputs.to(device).double()  # 将输入移动到 device
            labels = labels.to(device).long()  # 同时将标签移动到 device

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = total_loss / len(train_data_loader)
        epoch_train_acc = correct / total
        print(f'Epoch {epoch + 1}, train loss = {epoch_train_loss}, train acc = {epoch_train_acc}')

        if (epoch + 1) % 5 == 0:
            eval(model, test_data_loader, criterion, epoch)
    #             _eval_single_class(model, test_data_loader, criterion, epoch)

    final_eval(model, test_data_loader, criterion, filename)

    print('Finished Training and testing')

def main():
    # 加载数据
    real_loader,real_loader_small, test_real_loader,syn_loader,mixed_loader,=get_data_loaders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义模型、损失和优化器
    ECG_model=ECG_Net()
    ECG_model.double()
    ECG_model = ECG_model.to(device)  # 将模型移到 GPU（如果可用）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ECG_model.parameters(), lr=0.0005, momentum=0.9)

    # 训练并评估模型
    train(ECG_model, real_loader, test_real_loader, epochs=100, criterion=criterion, optimizer=optimizer,
          filename='real_data')
    #train(ECG_model, syn_loader, test_real_loader, 1, criterion, optimizer, filename='synthetic_data')

    #train(ECG_model, real_loader_small, test_real_loader, 1, criterion, optimizer, filename='real_data_small')
    #train(ECG_model, mixed_loader, test_real_loader, 1, criterion, optimizer, filename='mixed_data')
    plot_tsne(ECG_model, test_real_loader,device,filename='real_data')



if __name__ == "__main__":
    main()