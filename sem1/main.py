import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 1: Load the dataset
file_path = 'data/Seminars_1_Group_4.csv'  # Update the path as needed
data = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Separate features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column (class)

# Normalize features (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode string labels to numeric using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Converts labels to integers
y_categorical = to_categorical(y_encoded)   # Converts integers to one-hot encoding

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.15, random_state=42, stratify=y_encoded
)

# Output shapes for verification
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Step 4: Build the ANN Model
model = Sequential([
    Dense(24, activation='sigmoid', input_shape=(X_train.shape[1],)),
    Dense(y_train.shape[1], activation='softmax')  # Output layer
])

# Step 5: Compile the Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(X_train, y_train, 
                    validation_split=0.15, 
                    epochs=20, 
                    batch_size=32, 
                    verbose=1)

# Step 7: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 8: Visualize Training History
# Create a directory to save the plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig("plots/training_vs_validation_accuracy.png")  # Save plot
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig("plots/training_vs_validation_loss.png")  # Save plot
plt.show()

# Step 9: Create and Save Confusion Matrix
# Predict on the test set
y_pred = np.argmax(model.predict(X_test), axis=1)  # Convert one-hot to class labels
y_true = np.argmax(y_test, axis=1)  

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.savefig("plots/confusion_matrix.png")  # Save confusion matrix
plt.show()
