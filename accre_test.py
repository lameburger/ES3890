import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = model.to(device)  # Move model to GPU

dummy_input = torch.randn(1, 3, 224, 224).to(device)

model.eval()
with torch.no_grad():
    output = model(dummy_input)

print("Model output shape:", output.shape)
