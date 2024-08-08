#vgg16, adam, 80% train, 40 epok

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

IMAGES_PER_VIDEO = 20

def load_data_from_h5(filename):
    with h5py.File(filename, 'r') as file:
        data = file['data'][:]
        labels = file['labels'][:]
    
    data_batches = [data[i:i + IMAGES_PER_VIDEO] for i in range(0, len(data), IMAGES_PER_VIDEO)]
    label_batches = [labels[i] for i in range(0, len(labels), IMAGES_PER_VIDEO)]
    
    return np.array(data_batches), np.array(label_batches)

train_data, train_labels = load_data_from_h5('train_data.h5')
test_data, test_labels = load_data_from_h5('test_data.h5')

model = Sequential([
    LSTM(512, input_shape=(IMAGES_PER_VIDEO, 4096)),
    Dense(1024, activation='relu'),
    Dense(50, activation='sigmoid'),
    Dense(2, activation='softmax')
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

split_index = int(0.8 * len(train_data))
history = model.fit(train_data[:split_index], train_labels[:split_index], epochs=40,
                    validation_data=(train_data[split_index:], train_labels[split_index:]),
                    batch_size=100, verbose=2)

model.save('VGG16_LSTM_model.keras')

test_results = model.evaluate(test_data, test_labels)
for metric_name, metric_value in zip(model.metrics_names, test_results):
    print(f"{metric_name}: {metric_value}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('accuracy.eps', format='eps', dpi=1000)
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('loss.eps', format='eps', dpi=1000)
plt.show()

predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

conf_matrix = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Violence', 'Violence'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
