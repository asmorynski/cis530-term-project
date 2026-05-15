import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN_Waste
from statsmodels.stats.contingency_tables import mcnemar

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

correct_both = 0
correct_cnn = 0
correct_bsln = 0
correct_neither = 0
bsln_prediction = 3 #paper class

##evaluate
correct, total = 0, 0
with torch.no_grad():
	for img, lbl in test_loader:
		outputs = model(img)
		_, predicted = torch.max(outputs.data, 1)
		total += lbl.size(0)
		correct += (predicted == lbl).sum().item()

		#mcnemar check
		for i in range(len(lbl)):
			true_lbl = lbl[i].item()
			cnn_prediction = predicted[i].item()
			cnn = (cnn_prediction == true_lbl)
			base = (bsln_prediction == true_lbl)

			if cnn and base:
				correct_both += 1
			elif cnn and not base:
				correct_cnn += 1
			elif not cnn and base:
				correct_bsln += 1
			else: 
				correct_neither += 1

print(f"Accuracy: {(correct/total)*100:.2f}%") #Test accuracy

#McNemar Stuff
cont_table = [[correct_both, correct_bsln], [correct_cnn, correct_neither]]

#print table
print(f"Both: {correct_both} | Baseline: {correct_bsln} ")
print(f"CNN: {correct_cnn} | Neither: {correct_neither} ")

result = mcnemar(cont_table, exact=True)
print(f"P-Value: {result.pvalue}")