import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN_Waste

transform = transforms.Compose([
	transforms.Resize((128, 128)),
	transforms.ToTensor()
])

complete_data = datasets.ImageFolder(root="dataset", transform=transform)
train_size = int(.8 * len(complete_data))
test_size = len(complete_data) - train_size
_, test_data = torch.utils.data.random_split(complete_data, [train_size, test_size])

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print("Data loaded")

##load saved
model = CNN_Waste()
model.load_state_dict(torch.load("cnn_waste.pth"))
model.eval()

##evaluate
correct, total = 0, 0
with torch.no_grad():
	for img, lbl in test_loader:
		outputs = model(img)
		_, predicted = torch.max(outputs.data, 1)
		total += lbl.size(0)
		correct += (predicted == lbl).sum().item()

print(f"Accuracy: {(correct/total)*100:.2f}%")