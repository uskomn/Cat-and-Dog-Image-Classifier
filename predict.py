import torch
import torch.nn as nn
from model import CNN
from PIL import Image
import torchvision
import torchvision.transforms as transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CNN()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()  # 进入评估模式



transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 添加 batch 维度
    return image.to(device)

image_cat_path = "./data/predict/cat.jpg"
image_dog_path="./data/predict/dog.jpg"
image_cat = preprocess_image(image_cat_path)
image_dog = preprocess_image(image_dog_path)

with torch.no_grad():
    output = model(image_dog)
    prediction = (output.item() > 0.5)
    print(f"预测类别: {prediction}")

classes=['cats', 'dogs']
predicted_class = classes[int(prediction)]
print(f"预测类别名称: {predicted_class}")

