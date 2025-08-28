import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
from PIL import Image, ImageTk
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

# For now use ImageNet weights (test Grad-CAM)
# Later replace with: model.load_state_dict(torch.load("chexnet.pth", map_location="cpu"))
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 14)  # 14 classes like CheXNet
model.eval()


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# Grad-CAM hooks
features = None
gradients = None

def save_features(module, input, output):
    global features
    features = output

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Hook into last dense block (best layer for Grad-CAM in DenseNet121)
last_conv = model.features.denseblock4
last_conv.register_forward_hook(save_features)
last_conv.register_backward_hook(save_gradients)


def generate_gradcam(img_path):
    global features, gradients

    # Load image
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax().item()

    # Backward pass
    model.zero_grad()
    score = output[0, pred_class]
    score.backward()

    # Compute Grad-CAM
    weights = gradients.mean(dim=(2,3), keepdim=True)
    cam = (weights * features).sum(dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)

    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    return img, Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

# Tkinter GUI
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files","*.png;*.jpg;*.jpeg")])
    if file_path:
        original, gradcam = generate_gradcam(file_path)

        # Convert for Tkinter
        orig_tk = ImageTk.PhotoImage(original.resize((300,300)))
        grad_tk = ImageTk.PhotoImage(gradcam.resize((300,300)))

        orig_label.config(image=orig_tk)
        orig_label.image = orig_tk
        grad_label.config(image=grad_tk)
        grad_label.image = grad_tk

# Main window
root = tk.Tk()
root.title("CheXNet Grad-CAM")

btn = tk.Button(root, text="Select X-ray", command=select_image)
btn.pack()

frame = tk.Frame(root)
frame.pack()

orig_label = tk.Label(frame)
orig_label.pack(side="left", padx=10, pady=10)

grad_label = tk.Label(frame)
grad_label.pack(side="right", padx=10, pady=10)

root.mainloop()
