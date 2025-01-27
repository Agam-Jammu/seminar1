import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = 'Seminars_1_Group_4.csv'  # Update this path if needed
dataset = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset Overview:")
print(dataset.head())
print("\nDataset Info:")
print(dataset.info())
print("\nSummary Statistics:")
print(dataset.describe())

# Separate features and target
X = dataset.drop(columns=['class'])  # Features
y = dataset['class']  # Target

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nClasses:", label_encoder.classes_)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Output shapes for verification
print("\nShapes:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
