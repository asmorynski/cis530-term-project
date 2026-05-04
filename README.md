# cis530-term-project
CIS 530 Term Project - AI Waste Sorter

Performance:
*Baseline Accuracy:* 23.67%
*Model Accuracy:* ~75.00%

Structure:
'model.py': CNN model architecture
'train.py': Training script - Takes about 10 minutes
'evaluate.py': Evaluates from pre-trained model "cnn_waste.pth"
'requirements.txt': Python packages required to run the project

How to run:

Option 1: Retraining -
Run 'python train.py'

Option 2: Using pre-trained weights - 
Download 'cnn_waste.pth' from google drive link: https://drive.google.com/file/d/1B7FBkWTveO_KP39wPymnUvaOqoQTIXX8/view?usp=sharing - 
Place 'cnn_waste.pth' in root directory - 
Run 'python evaluate.py' - 
