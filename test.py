import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mymodel import MyCIFAR10Net

# Data loading (test set)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCIFAR10Net(num_classes=10, use_batchnorm=True, use_dropout=True, activation='relu').to(device)
model.load_state_dict(torch.load('model/best_model_1.pth', map_location=device))
model.eval()

# Evaluate
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
