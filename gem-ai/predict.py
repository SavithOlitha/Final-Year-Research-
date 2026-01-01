import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image

IMG_SIZE = 224

def load_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load("gem_model.pt", map_location="cpu"))
    model.eval()
    return model

def predict(image_path):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
        real_p = float(prob[0])
        syn_p  = float(prob[1])

    label = "REAL" if real_p >= syn_p else "SYNTHETIC"
    print("Result:", label)
    print("Real prob:", real_p, "| Synthetic prob:", syn_p)

if __name__ == "__main__":
    # example: python predict.py composites/real/stone001.jpg
    import sys
    predict(sys.argv[1])
