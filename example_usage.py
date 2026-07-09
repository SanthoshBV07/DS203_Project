# example_usage.py
"""
The input folder contains the images for whcih the predictions are to be made. The output images are stored in a new folder named predictions and the csv file 
predictions.csv is created in the current directory containing the predicted labels.

Ensure that the files block_module.py and best_model.pkl are present in the same directory 

Required Libraries numpy, pandas, opencv-python, scikit-learn, imbalanced-learn, tqdm

"""
import pickle
from block_module import BlockClassifier

# Load the trained model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Wrap in classifier
model = BlockClassifier(best_model)

# Predict a folder of images
predictions = model.predict_folder("images_transformed","traintest.csv","traintest") #(input_folder, output csv name, output folder name)

print(predictions.head())
