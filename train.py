import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN_Waste

transform = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(10),
	transforms.ToTensor()
])

complete_data = datasets.ImageFolder(root="dataset", transform=transform)
train_size = int(.8 * len(complete_data))
test_size = len(complete_data) - train_size
train_data, test_data = torch.utils.data.random_split(complete_data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print("Data loaded")

##setup
model = CNN_Waste() ##model from model.py
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

##training
epochs = 15
print("Starting training")
for epoch in range(1, epochs+1):
	model.train()
	for img, lbl in train_loader:
		optimizer.zero_grad()
		loss = criteria(model(img), lbl)
		loss.backward()
		optimizer.step()
	print(f"Epoch {epoch} done. ")

##evaluation
model.eval()
correct, total = 0, 0

with torch.no_grad():
	for img, lbl in test_loader:
		outputs = model(img)
		_, predicted = torch.max(outputs.data, 1)
		total += lbl.size(0)
		correct += (predicted == lbl).sum().item()

print(f"Accuracy: {(correct/total)*100:.2f}%")