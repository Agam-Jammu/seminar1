import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
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

# Step 3: Feature Importance Analysis using ExtraTreesClassifier
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_scaled, y_encoded)

# Get feature importances
feature_importances = et_model.feature_importances_
feature_names = data.columns[:-1]

# Display feature importances
sorted_indices = np.argsort(feature_importances)[::-1]
print("Feature Importance Ranking:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {feature_importances[i]:.4f}")

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align="center")
plt.xticks(range(len(feature_importances)), feature_names[sorted_indices], rotation=45, ha='right')
plt.title('Feature Importance')
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/feature_importance.png")
plt.show()

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.15, random_state=42, stratify=y_encoded
)

# Step 5: Build the ANN Model
model = Sequential([
    Dense(24, activation='sigmoid', input_shape=(X_train.shape[1],)),
    Dense(y_train.shape[1], activation='softmax')  # Output layer
])

# Step 6: Compile the Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 7: Train the Model
history = model.fit(X_train, y_train, 
                    validation_split=0.15, 
                    epochs=20, 
                    batch_size=32, 
                    verbose=1)

# Step 8: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy (ANN): {test_accuracy:.2f}")

# Step 9: Visualize Training History
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig("plots/training_vs_validation_accuracy.png")
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig("plots/training_vs_validation_loss.png")
plt.show()

# Step 10: Create and Save Confusion Matrix (ANN)
# Predict on the test set
y_pred_ann = np.argmax(model.predict(X_test), axis=1)  # Convert one-hot to class labels
y_true = np.argmax(y_test, axis=1)  

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_ann)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.savefig("plots/confusion_matrix.png")  # Save confusion matrix
plt.show()
