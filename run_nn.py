import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from nn_model import PricePredictionNN
from data_utils import *

# Initialize dataset
original_data = pd.read_csv('car-data/car_price_prediction.csv')

processed_data = preprocess(original_data)
OHE_processed_data = OHE_data(processed_data)
X_train, y_train, X_test, y_test = split_data(OHE_processed_data) 

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert boolean columns to numeric (1 for True, 0 for False)
X_train = X_train.astype('float32')
X_test = X_train.astype('float32')

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshaped to be a column vector
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# Define hyperparameter search ranges
lr_values = [0.0001, 0.001, 0.01]  # Learning rates
batch_size_values = [32, 64, 128]  # Batch sizes
epochs_values = [50, 100, 150, 300]  # Number of epochs
decay_values = [0, 0.1, 0.5, 1, 2, 3]  # ridge regression lambda

# # Run hyperparameter tuning
# best_model, best_params, best_rmse = hyperparameter_tuning(lr_values, batch_size_values, epochs_values, decay_values, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# # Output the best hyperparameters and RMSE
# print(f"Best Hyperparameters: Learning Rate = {best_params[0]}, Batch Size = {best_params[1]}, Epochs = {best_params[2]}, Decay={best_params[3]}")
# print(f"Best RMSE: {best_rmse:.4f}")

# Initialize the model
model = PricePredictionNN(input_dim=X_train_tensor.shape[1])

# Train the model
model.train_model(X_train_tensor, y_train_tensor, epochs=150, lr=0.001, batch_size=64, decay=0.5)

# Evaluate the model and get test RMSE
print("Evaluating model...")
test_rmse = model.eval_model(X_test_tensor, y_test_tensor)



# Best Hyperparameters: Learning Rate = 0.01, Batch Size = 64, Epochs = 150
# Best RMSE: 10977.9717