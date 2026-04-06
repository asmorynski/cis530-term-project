import random

def class_baseline(total_img, class_dist):
	print("-Naive Baseline-")
	majority_class = max(class_dist, key=class_dist.get)
	majority_count = class_dist[majority_class]

	acc = (majority_count/total_img) * 100
	print(f"Total Images: {total_img}")
	print(f"Most Freq. Class: {majority_class}")
	print(f"Baseline Accuracy: {acc:.2f}")


dataset_dist = {
	"cardboard": 393,
	"glass": 491,
	"metal": 400,
	"paper": 584,
	"plastic": 472,
	"trash": 127
}

total_img = sum(dataset_dist.values())
class_baseline(total_img, dataset_dist)