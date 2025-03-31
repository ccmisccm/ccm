
from DataLoader_2D_uav import mitbih_train, mitbih_test
from make_synDataLoader_1D_uav import mixed_mitbih,syn_mitbih
from classification_model import *
from torch.utils import data
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')  # 或者使用其他可用的后端，如 'Agg', 'Qt5Agg'
import matplotlib.pyplot as plt



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

        for i, data in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.double()
            labels = labels.long()

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
    # 定义模型、损失和优化器
    ECG_model=ECG_Net()
    ECG_model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ECG_model.parameters(), lr=0.0005, momentum=0.9)

    # 训练并评估模型
    train(ECG_model, real_loader, test_real_loader, epochs=1, criterion=criterion, optimizer=optimizer,
          filename='real_data')
    train(ECG_model, syn_loader, test_real_loader, 1, criterion, optimizer, filename='synthetic_data')

    train(ECG_model, real_loader_small, test_real_loader, 1, criterion, optimizer, filename='real_data_small')
    train(ECG_model, mixed_loader, test_real_loader, 1, criterion, optimizer, filename='mixed_data')



if __name__ == "__main__":
    main()