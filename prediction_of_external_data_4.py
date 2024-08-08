# Please make sure "train_model_2.py" and "feature_extraction_of_external_data_3.py" must be run befor this code. 
# After compilation of these 2, there will be outputs as "VGG16_LSTM_model.keras" and "externaldata.h5"
# "VGG16_LSTM_model.keras" this is the model that we have trained. In here we will load the model and the external data and will evaluate our model with external data

# Please enter the external dataset file path in line 16


import h5py
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMAGES_PER_VIDEO = 20
IMG_DIMS = (224, 224)
DATA_DIR = "/Users/nasir/Downloads/Archive 2/ExternalData" # Verisetinden ayıracağınız videoları tek klasörde topladıktan sonra buraya dosya yolunu giriniz

def load_data_from_h5(filename):
    with h5py.File(filename, 'r') as file:
        data = file['data'][:]
        labels = file['labels'][:]
        filenames = file['filenames'][:]
    
    data_batches = [data[i:i + IMAGES_PER_VIDEO] for i in range(0, len(data), IMAGES_PER_VIDEO)]
    label_batches = [labels[i] for i in range(0, len(labels), IMAGES_PER_VIDEO)]
    filename_batches = [filenames[i] for i in range(0, len(filenames), IMAGES_PER_VIDEO)]
    
    return np.array(data_batches), np.array(label_batches), np.array(filename_batches)

# Load the data
test_data, test_labels, test_filenames = load_data_from_h5('externaldata.h5')

# Load the model
model = load_model('VGG16_LSTM_model.keras')

# Print test labels with headers
print("Test Labels (0: Violence, 1: No Violence):")
print(tabulate(test_labels, headers=["Violence", "No Violence"], tablefmt="grid"))

# Predict
predictions = model.predict(test_data)

# Print predictions with headers
print("\nPredictions:")
print(tabulate(predictions, headers=["Violence", "No Violence"], tablefmt="grid"))

# Interpret predictions
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Collect results for table
results = []

for i in range(len(test_data)):
    is_correct = predicted_classes[i] == true_classes[i]
    video_name = test_filenames[i].decode('utf-8')
    
    predicted_label = 'Violence' if predicted_classes[i] == 0 else 'No Violence'
    true_label = 'Violence' if true_classes[i] == 0 else 'No Violence'
    
    results.append([video_name, predicted_label, true_label, is_correct])

# Display results in a table
print("\nResults:")
print(tabulate(results, headers=["Video Name", "Predicted Label", "True Label", "Correct"], tablefmt="grid"))

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Violence", "No Violence"], yticklabels=["Violence", "No Violence"])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=["Violence", "No Violence"]))
