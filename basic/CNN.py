# CNN implementation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

# Prepare the dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST (
  root="./data",
  train=True,
  download=True,
  transform=transform
)

# print(len(train_dataset))
# image, label = train_dataset[0]
# print(image.shape)
# print(label)
# 60000
# torch.Size([1, 28, 28])
# 5

# Use dataloader to load data
train_loader = DataLoader(
  dataset=train_dataset,
  batch_size=32,
  shuffle=True
)

# The output of Dataloader is several batches, the first dimension of each batch is batch size
# for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
#     print(f"Batch {batch_idx}: X={X_batch.shape}, y={y_batch.shape}")

# Batch 0: X=torch.Size([32, 1, 28, 28]), y=torch.Size([32])
# Batch 1: X=torch.Size([32, 1, 28, 28]), y=torch.Size([32])


# Define the model
# Net = nn.Sequential(
#   nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # (1, 28, 28) -> (32, 28, 28)
#   nn.ReLU(),
#   nn.MaxPool2d(2), #(32, 28, 28) -> (32, 14, 14)
#   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (32, 14, 14) -> (64, 14, 14)
#   nn.ReLU(),
#   nn.MaxPool2d(2), # (64, 14, 14) -> (64, 7, 7)
#   nn.Flatten(),
#   nn.Linear(64 * 7 * 7, 128),
#   nn.ReLU(),
#   nn.Linear(128 , 10)
# )

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = x.view(x.size(0), -1) #flatten
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# model training
model = Net()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()

epoch = 10
for i in range(epoch):
  for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
    # Forward
    output = model(X_batch)
    loss = criterion(output, y_batch)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if batch_idx % 100 == 0:
      print(f"Epoch {i}, Batch {batch_idx}, Loss: {loss.item():.4f}")

