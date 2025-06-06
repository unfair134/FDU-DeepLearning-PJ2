import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from mymodel import MyCIFAR10Net
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(trainset))
valid_size = len(trainset) - train_size
train_subset, valid_subset = random_split(trainset, [train_size, valid_size])
trainloader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
valloader = DataLoader(valid_subset, batch_size=128, shuffle=False, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# 2. Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCIFAR10Net(num_classes=10, use_batchnorm=True, use_dropout=True, activation='leakyrelu').to(device)
loss_fn = nn.CrossEntropyLoss()  # Try different loss functions here
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Try different optimizers here
# 切换优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)

epoch_losses = []

def train(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = loss_fn(outputs, labels)

            l2_lambda = 1e-4  # L2正则化强度
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss = loss_fn(outputs, labels) + l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss/len(trainloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)
        validate()
    # Save best model if needed
    import time
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    path=f'model/best_model_{time_stamp}.pth'
    torch.save(model.state_dict(), path)
    # 保存loss曲线
    plt.figure()
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.savefig('loss_curve.png')
    plt.close()

def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

def save_all_conv_filters(model, filename='all_filters.png', layer='conv1'):
    if layer == 'conv1':
        filters = model.conv1.weight.data.clone().cpu()
    elif layer == 'conv2':
        filters = model.conv2.weight.data.clone().cpu()
    else:
        raise ValueError("layer must be 'conv1' or 'conv2'")
    num_filters = filters.shape[0]
    ncols = 8
    nrows = (num_filters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    for i in range(num_filters):
        r, c = divmod(i, ncols)
        f = filters[i]
        f_min, f_max = f.min(), f.max()
        f = (f - f_min) / (f_max - f_min)
        if f.shape[0] == 3:  # RGB
            axes[r, c].imshow(f.permute(1, 2, 0))
        else:  # 单通道
            axes[r, c].imshow(f[0], cmap='gray')
        axes[r, c].axis('off')
    for i in range(num_filters, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_all_feature_maps(model, image, filename='feature_maps.png', after='conv1'):
    model.eval()
    with torch.no_grad():
        x = image.unsqueeze(0).to(next(model.parameters()).device)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model._activate(x)
        if after == 'conv2':
            x = model.pool(x)
            x = model.conv2(x)
            x = model.bn2(x)
            x = model._activate(x)
        feature_maps = x.cpu().squeeze(0)
        num_maps = feature_maps.shape[0]
        ncols = 8
        nrows = (num_maps + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
        for i in range(num_maps):
            r, c = divmod(i, ncols)
            fmap = feature_maps[i]
            fmap_min, fmap_max = fmap.min(), fmap.max()
            fmap = (fmap - fmap_min) / (fmap_max - fmap_min)
            axes[r, c].imshow(fmap, cmap='viridis')
            axes[r, c].axis('off')
        for i in range(num_maps, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def plot_loss_landscape(model, dataloader, loss_fn, steps=20, alpha=0.5):
    w = model.fc1.weight.data.clone()
    direction1 = torch.randn_like(w)
    direction2 = torch.randn_like(w)
    losses = np.zeros((steps, steps))
    device = next(model.parameters()).device
    for i, a in enumerate(np.linspace(-alpha, alpha, steps)):
        for j, b in enumerate(np.linspace(-alpha, alpha, steps)):
            model.fc1.weight.data = w + a * direction1 + b * direction2
            total_loss = 0
            count = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                count += 1
                if count > 2:  # 只用少量batch加速
                    break
            losses[i, j] = total_loss / count
    model.fc1.weight.data = w  # 恢复原权重
    plt.figure(figsize=(6,5))
    plt.contourf(losses, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Loss Landscape (fc1 weight)')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.savefig('loss_landscape.png')
    plt.close()

if __name__ == "__main__":
    train(num_epochs=10)
    # model.eval()
    # save_conv1_filters(model)

    # 加载已有模型
    # model.load_state_dict(torch.load('best_model.pth', map_location=device))
    # model.eval()

    # # 保存所有卷积核
    # save_all_conv_filters(model, filename='all_filters_conv1.png', layer='conv1')
    # save_all_conv_filters(model, filename='all_filters_conv2.png', layer='conv2')

    # # 取一张验证集图片
    # sample_img, _ = next(iter(valloader))

    # # 可视化feature map
    # visualize_all_feature_maps(model, sample_img[0], filename='feature_maps_conv1.png', after='conv1')
    # visualize_all_feature_maps(model, sample_img[0], filename='feature_maps_conv2.png', after='conv2')

    # # Loss landscape visualization
    # plot_loss_landscape(model, valloader, loss_fn)

    # Evaluate on test set after training
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print(f'Test Error: {100 - 100 * correct / total:.2f}%')