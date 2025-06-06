import torch
import matplotlib.pyplot as plt
from mymodel import MyCIFAR10Net

# Example: Visualize first conv layer filters

def visualize_filters(model_path='best_model.pth', save_path='filters.png'):
    model = MyCIFAR10Net(num_classes=10)
    model.load_state_dict(torch.load(model_path))
    conv1_weights = model.conv1.weight.data.cpu()
    num_filters = conv1_weights.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters*2, 2))
    for i in range(num_filters):
        ax = axes[i]
        # Normalize to [0,1] for visualization
        w = conv1_weights[i]
        w = (w - w.min()) / (w.max() - w.min())
        ax.imshow(w.permute(1,2,0))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# You can add more visualization functions (e.g., loss landscape, feature maps, etc.)

if __name__ == "__main__":
    visualize_filters(save_path='my_filters.png')
