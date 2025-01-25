import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
file_path = '../data/Seminars_1_Group_4.csv'  # Update the path as needed
data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Separate features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column (class)

# Normalize features (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the target variable (class column)
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y
)

# Output shapes for verification
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Save processed data for reuse
pd.DataFrame(X_train).to_csv('../data/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('../data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)
