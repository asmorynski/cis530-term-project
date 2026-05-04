import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.ToTensor()
])

complete_data = datasets.ImageFolder(root="dataset", transform=transform)
train_size = int(.8 * len(complete_data))
test_size = len(complete_data) - train_size
train_data, test_data = torch.utils.data.random_split(complete_data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print("Data loaded")