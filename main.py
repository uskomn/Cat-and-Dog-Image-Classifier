import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNN
from torchvision import models

transform=transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset=torchvision.datasets.ImageFolder(root='./data/training_set',transform=transform)
test_dataset=torchvision.datasets.ImageFolder(root='./data/test_set',transform=transform)

classes=train_dataset.classes
print(classes)
train_loader=DataLoader(dataset=train_dataset,batch_size=8,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=8,shuffle=True)



if torch.cuda.is_available():
    print("gpu")
else:
    print("cpu")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN().to(device)
criterion=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

epochs=10
train_losses=[]
test_losses=[]

for epoch in range(epochs):
    model.train()
    running_loss=0
    train_correct=0
    train_total=0
    for images, labels in train_loader:
        images,labels=images.to(device),labels.float().to(device).view(-1,1)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        predicted=(outputs>0.5).float()
        train_correct+=(predicted==labels).sum().item()
        train_total+=labels.size(0)

    model.eval()
    test_loss=0.0
    test_correct=0
    test_total=0
    with torch.no_grad():
        for images, labels in test_loader:
            images,labels=images.to(device),labels.float().to(device).view(-1,1)
            outputs=model(images)
            loss=criterion(outputs,labels)
            test_loss+=loss.item()

            predicted=(outputs>0.5).float()
            test_correct+=(predicted==labels).sum().item()
            test_total+=labels.size(0)
    train_losses.append(running_loss/len(train_loader))
    test_losses.append(test_loss/len(test_loader))

    train_accuracy = 100 * train_correct / train_total
    test_accuracy = 100 * test_correct / test_total

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader):.4f}, "
          f"Test Loss: {test_loss / len(test_loader):.4f}, "
          f"Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")



torch.save(model.state_dict(), 'model.pth')
print("~结束训练")

