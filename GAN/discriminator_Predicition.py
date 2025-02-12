import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Define the Discriminator class (must match the original architecture!)
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Load CSV dataset (same preprocessing as in training)
real_df = pd.read_csv("data/CD.csv")  # Original dataset
generated_df = pd.read_csv("data/generated_data.csv")  # GAN-generated dataset

# Identify columns
categorical_columns = real_df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = real_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Load pre-fitted scalers (to match training)
num_scaler = MinMaxScaler()
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

df_num = real_df[numerical_columns]
df_cat = real_df[categorical_columns]

num_scaled = num_scaler.fit_transform(df_num)
cat_encoded = cat_encoder.fit_transform(df_cat)

# Transform both real and generated data
real_processed = np.hstack((num_scaled, cat_encoded))
real_data = torch.tensor(real_processed, dtype=torch.float32)

gen_processed = np.hstack((num_scaler.transform(generated_df[numerical_columns]), cat_encoder.transform(generated_df[categorical_columns])))
generated_data = torch.tensor(gen_processed, dtype=torch.float32)

# Load the trained Discriminator
input_dim = real_data.shape[1]  # Must match training input dimensions
discriminator = Discriminator(input_dim)
discriminator.load_state_dict(torch.load("models/discriminator.pth"))  # Load saved model
discriminator.eval()  # Set to evaluation mode

# Function to classify a row as real or generated
def classify_row(row_tensor):
    with torch.no_grad():
        prediction = discriminator(row_tensor.unsqueeze(0))  # Add batch dimension
        confidence = prediction.item()
        return "Real" if confidence > 0.5 else "Generated", confidence

# Test the discriminator on real and generated rows
print("\n🔍 **Discriminator Results** 🔍")

for i, row in enumerate(real_data[:5]):  # Check 5 real rows
    label, confidence = classify_row(row)
    print(f"Real Row {i + 1}: {label} (Confidence: {confidence:.4f})")

for i, row in enumerate(generated_data[:5]):  # Check 5 generated rows
    label, confidence = classify_row(row)
    
    print(f"Generated Row {i + 1}: {label} (Confidence: {confidence:.4f})")
