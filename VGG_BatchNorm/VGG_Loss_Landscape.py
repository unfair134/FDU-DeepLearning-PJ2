import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    #
    #
    #
    #
    ## --------------------
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = []
    train_accuracy_curve = []
    val_accuracy_curve = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        learning_curve.append(avg_loss)
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accuracy_curve.append(train_acc)
        val_accuracy_curve.append(val_acc)
        print(f'Epoch {epoch+1}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')
    return learning_curve, train_accuracy_curve, val_accuracy_curve

def train_stepwise(model, optimizer, criterion, train_loader, val_loader, epochs_n=10, save_prefix=''):
    model.to(device)
    model.train()
    step_losses = []
    step_grads = []
    for epoch in range(epochs_n):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            # 记录loss
            step_losses.append(loss.item())
            # 记录梯度范数
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            step_grads.append(total_norm)
            optimizer.step()
    # 保存loss和grad
    np.save(f'{save_prefix}_step_losses.npy', np.array(step_losses))
    np.save(f'{save_prefix}_step_grads.npy', np.array(step_grads))
    return step_losses, step_grads

# Train your model
# feel free to modify
epo = 5
loss_save_path = ''
# grad_save_path = ''

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
loss, train_acc, val_acc = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
# np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
#
#
#
#
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    ## --------------------
    # Add your code
    #
    #
    #
    #
    ## --------------------
    pass

if __name__ == "__main__":
    epo = 10
    set_random_seeds(seed_value=2020, device=device)

    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    criterion = nn.CrossEntropyLoss()
    all_loss_a = []
    all_loss_bn = []
    all_grad_a = []
    all_grad_bn = []
    for lr in learning_rates:
        # VGG-A
        model_a = VGG_A()
        optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr)
        prefix_a = f'models/vgg_a_lr{lr}'
        step_losses_a, step_grads_a = train_stepwise(model_a, optimizer_a, criterion, train_loader, val_loader, epochs_n=epo, save_prefix=prefix_a)
        torch.save(model_a.state_dict(), f'{prefix_a}.pth')
        all_loss_a.append(step_losses_a)
        all_grad_a.append(step_grads_a)

        # VGG-A-BN
        model_bn = VGG_A_BatchNorm()
        optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
        prefix_bn = f'models/vgg_bn_lr{lr}'
        step_losses_bn, step_grads_bn = train_stepwise(model_bn, optimizer_bn, criterion, train_loader, val_loader, epochs_n=epo, save_prefix=prefix_bn)
        torch.save(model_bn.state_dict(), f'{prefix_bn}.pth')
        all_loss_bn.append(step_losses_bn)
        all_grad_bn.append(step_grads_bn)

    # 分别为每个learning rate单独画图
    for lr in learning_rates:
        # 读取loss和grad
        loss_a = np.load(f'models/vgg_a_lr{lr}_step_losses.npy')
        loss_bn = np.load(f'models/vgg_bn_lr{lr}_step_losses.npy')
        grad_a = np.load(f'models/vgg_a_lr{lr}_step_grads.npy')
        grad_bn = np.load(f'models/vgg_bn_lr{lr}_step_grads.npy')
        steps = np.arange(1, min(len(loss_a), len(loss_bn)) + 1)

        # Loss对比
        plt.figure(figsize=(8,5))
        plt.plot(steps, loss_a[:len(steps)], 'r-', label='VGG-A (no BN)')
        plt.plot(steps, loss_bn[:len(steps)], 'b-', label='VGG-A (with BN)')
        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title(f'Loss Curve (lr={lr})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'vgg_loss_curve_lr{lr}.png')
        plt.close()

        # 梯度范数对比
        plt.figure(figsize=(8,5))
        plt.plot(steps, grad_a[:len(steps)], 'r-', label='VGG-A (no BN)')
        plt.plot(steps, grad_bn[:len(steps)], 'b-', label='VGG-A (with BN)')
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title(f'Gradient Norm Curve (lr={lr})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'vgg_grad_norm_curve_lr{lr}.png')
        plt.close()

        # 最大梯度差
        grad_diff_a = np.abs(np.diff(grad_a[:len(steps)]))
        grad_diff_bn = np.abs(np.diff(grad_bn[:len(steps)]))
        plt.figure(figsize=(8,5))
        plt.plot(steps[1:], grad_diff_a, 'r-', label='VGG-A (no BN)')
        plt.plot(steps[1:], grad_diff_bn, 'b-', label='VGG-A (with BN)')
        plt.xlabel('Step')
        plt.ylabel('Gradient Difference')
        plt.title(f'Max Gradient Difference (lr={lr})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'vgg_max_grad_diff_lr{lr}.png')
        plt.close()