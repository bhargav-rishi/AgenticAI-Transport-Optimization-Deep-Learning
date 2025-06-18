import pandas as pd

# Load the dataset (Replace 'your_file.csv' with actual file name)
file_path = "/content/drive/MyDrive/Colab Notebooks/FAF4_Regional.csv"
df = pd.read_csv(file_path)

# Select only the relevant columns for our AI model
df = df[["dms_orig", "dms_dest", "dms_mode", "tons_2012", "value_2012", "tmiles_2012", "trade_type"]]

# Filter for domestic shipments only (trade_type == 1)
df = df[df["trade_type"] == 1]

# Drop trade_type column since it's now redundant
df.drop(columns=["trade_type"], inplace=True)

# Display cleaned dataset
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Cleaned FAF Dataset", dataframe=df)
df.head(5)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

# 🚀 **Step 1: Data Preprocessing & Balancing**
# Convert transport mode from numeric to categorical labels
mode_mapping = {
    1: "Truck",
    2: "Rail",
    3: "Water",
    4: "Air",
    5: "Multiple_Modes_and_Mail",
    6: "Pipeline",
    7: "Other_and_Unknown"
}
df["dms_mode"] = df["dms_mode"].map(mode_mapping)

# One-hot encode transport mode
df = pd.get_dummies(df, columns=["dms_mode"], prefix="dms_mode", dtype=int)

# Balance dataset to ensure all transport modes have equal representation
mode_counts = df.filter(like="dms_mode_").sum()
min_samples = mode_counts.min()

balanced_df_list = []
for mode in mode_counts.index:
    mode_subset = df[df[mode] == 1]
    balanced_mode = resample(mode_subset, replace=True, n_samples=min_samples, random_state=42)
    balanced_df_list.append(balanced_mode)

df_balanced = pd.concat(balanced_df_list)

print("✅ Dataset balanced with equal transport mode representation!")

# 🚀 **Step 2: Feature Scaling**
# Use a separate scaler for shipment cost (`value_2012`)
scaler_value = MinMaxScaler()
df_balanced["value_2012"] = scaler_value.fit_transform(df_balanced[["value_2012"]])

# Normalize other numerical features
scaler_features = MinMaxScaler()
df_balanced[["tons_2012", "tmiles_2012"]] = scaler_features.fit_transform(df_balanced[["tons_2012", "tmiles_2012"]])

# 🚀 **Step 3: Split Data**
X = df_balanced.drop(columns=["value_2012"])
y = df_balanced["value_2012"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to float32 for memory efficiency
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# 🚀 **Step 4: Move Data to GPU Efficiently**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader with mini-batch processing
batch_size = 1024
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 🚀 **Step 5: Define Neural Network with L2 Regularization & Dropout**
class ShipmentNN(nn.Module):
    def __init__(self, input_size):
        super(ShipmentNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout to prevent overfitting

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Model
input_size = X_train.shape[1]
model = ShipmentNN(input_size).to(device)

# Initialize model & move to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 penalty added here

# 🚀 **Step 6: Train the Neural Network**
epochs = 400
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)  # Move batch to GPU

        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

print("✅ Training Completed with GPU Acceleration!")

# 🚀 **Step 7: Model Evaluation**
with torch.no_grad():
    test_losses = []
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        y_pred_test = model(batch_X)
        loss = criterion(y_pred_test, batch_y)
        test_losses.append(loss.item())

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"\n🔍 Final Test Loss: {avg_test_loss:.4f}")

