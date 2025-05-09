import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score


class PricePredictionNN(nn.Module):
   def __init__(self, input_dim):
      super(PricePredictionNN, self).__init__()
      self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer (64 nodes)
      self.fc2 = nn.Linear(64, 32)         # Second hidden layer (32 nodes)
      self.fc3 = nn.Linear(32, 1)          # Output layer (1 nodes)
      self.dropout = nn.Dropout(p=0.5) # 30! chance of dropout

   def forward(self, x):
      x = torch.relu(self.fc1(x))  # Activation function ReLU
      x = self.dropout(x)
      x = torch.relu(self.fc2(x))
      x = self.dropout(x)
      x = self.fc3(x)
      return x

   def train_model(self, X_train_tensor, y_train_tensor, epochs, lr, batch_size, decay):
      # Create a DataLoader for batch processing
      train_data = TensorDataset(X_train_tensor, y_train_tensor)
      train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

      # Set up loss function and optimizer
      loss_fn = nn.MSELoss()  # Mean Squared Error for regression
      optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=decay)  # Adam optimizer

      # Training the model
      for epoch in range(epochs):
         self.train()
         running_loss = 0.0
         
         for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
           
            # Forward pass
            predictions = self(X_batch)
            
            # Calculate loss
            loss = loss_fn(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()
         
         if (epoch+1) % 10 == 0:  # Print loss every 10 epochs
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
   
   def eval_model(self, X_test_tensor, y_test_tensor):
      # Evaluate on test set
      self.eval()
      with torch.no_grad():
         y_pred_log = self(X_test_tensor).view(-1)

      # Inverse log-transform: go back to original price scale
      y_pred = torch.expm1(y_pred_log)
      y_true = torch.expm1(y_test_tensor.view(-1))

      # Move to NumPy
      y_pred_np = y_pred.cpu().numpy()
      y_true_np = y_true.cpu().numpy()

      # Calculate RMSE
      test_rmse = np.sqrt(np.mean((y_pred_np - y_true_np) ** 2))
      print(f"Neural Network Test RMSE: {test_rmse:.2f}")

      # Calculate R²
      test_r2 = r2_score(y_true_np, y_pred_np)
      print(f"Neural Network Test R²: {test_r2:.4f}")

      return test_rmse, test_r2