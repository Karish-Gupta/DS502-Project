import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from nn_model import PricePredictionNN
from data_utils import *
from sklearn.model_selection import StratifiedKFold

''' Initialize dataset '''
original_data = pd.read_csv('car-data/car_price_prediction.csv')

processed_data = preprocess(original_data)
OHE_processed_data = OHE_data(processed_data)
X_train, y_train, X_test, y_test = split_data(OHE_processed_data) 

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = np.log1p(y_train)  # log(1 + y)
y_test = np.log1p(y_test)

# Convert boolean columns to numeric (1 for True, 0 for False)
X_train = X_train.astype('float32')
X_test = X_train.astype('float32')

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshaped to be a column vector
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


''' Hyperparameter tuning'''
# # Define hyperparameter search ranges
# lr_values = [0.0001, 0.001, 0.01]  # Learning rates
# batch_size_values = [32, 64, 128]  # Batch sizes
# epochs_values = [50, 100, 150, 300]  # Number of epochs
# decay_values = [0, 0.1, 0.5, 1, 2, 3]  # ridge regression lambda

# # Run hyperparameter tuning
# best_model, best_params, best_rmse = hyperparameter_tuning(lr_values, batch_size_values, epochs_values, decay_values, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# # Output the best hyperparameters and RMSE
# print(f"Best Hyperparameters: Learning Rate = {best_params[0]}, Batch Size = {best_params[1]}, Epochs = {best_params[2]}, Decay={best_params[3]}")
# print(f"Best RMSE: {best_rmse:.4f}")


# Best Hyperparameters: Learning Rate = 0.01, Batch Size = 64, Epochs = 150, Decay = 0
# Best RMSE: 10977.9717


''' K-Fold CV '''
# 📦 Define stratified K-Fold
price_bins = pd.qcut(y_train.values, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store RMSE and R² scores for each fold
rmse_scores = []
r2_scores = []

# K-Fold Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, price_bins)):
   print(f"\n Fold {fold+1}")

   # Split the data
   X_train_fold = X_train_tensor[train_idx]
   y_train_fold = y_train_tensor[train_idx]
   X_val_fold = X_train_tensor[val_idx]
   y_val_fold = y_train_tensor[val_idx]

   # Initialize a fresh model for each fold
   model = PricePredictionNN(input_dim=X_train_tensor.shape[1])
   
   # Train the model on this fold
   model.train_model(
      X_train_fold, 
      y_train_fold, 
      epochs=150, 
      lr=0.001, 
      batch_size=64, 
      decay=0
   )

   # Evaluate on validation fold
   val_rmse, val_r2 = model.eval_model(X_val_fold, y_val_fold)
   rmse_scores.append(val_rmse)
   r2_scores.append(val_r2)

# Summary
print("\n K-Fold Cross-Validation Results:")
print(f"🔹 Average RMSE: {np.mean(rmse_scores):.2f}")
print(f"🔹 Std Dev RMSE: {np.std(rmse_scores):.2f}")
print(f"🔹 Average R²: {np.mean(r2_scores):.4f}")
print(f"🔹 Std Dev R²: {np.std(r2_scores):.4f}")

# # Optional: Train final model on full training set
# final_model = PricePredictionNN(input_dim=X_train_tensor.shape[1])
# final_model.train_model(X_train_tensor, y_train_tensor, epochs=1, lr=0.001, batch_size=128, decay=0)

# # Test set evaluation
# print("\n Final Evaluation on Test Set:")
# test_rmse, test_r2 = final_model.eval_model(X_test_tensor, y_test_tensor)



#  K-Fold Cross-Validation Results:
# Average RMSE: 12582.52
