import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PricePredictionNN(nn.Module):
   def __init__(self, input_dim):
      super(PricePredictionNN, self).__init__()
      self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer (128 nodes)
      self.fc2 = nn.Linear(128, 64)         # Second hidden layer (64 nodes)
      self.fc3 = nn.Linear(64, 32)          # Third hidden layer (32 nodes)
      self.fc4 = nn.Linear(32, 1)           # Output layer (1 node for regression)
      self.dropout = nn.Dropout(p=0.3) # 30! chance of dropout

   def forward(self, x):
      x = torch.relu(self.fc1(x))  # Activation function ReLU
      x = self.dropout(x)
      x = torch.relu(self.fc2(x))
      x = self.dropout(x)
      x = torch.relu(self.fc3(x))
      x = self.fc4(x)
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
   
   def eval_model(self, X_test_tensor,  y_test_tensor,):
      # Evaluate on test set
      self.eval()
      with torch.no_grad():
         y_pred_test = self(X_test_tensor)
      
      y_pred_test = y_pred_test.view(-1)  # Flatten the predictions to match the shape of y_test_tensor

      # Calculate RMSE
      test_rmse = torch.sqrt(torch.mean((y_pred_test - y_test_tensor) ** 2)).item()
      print(f"Neural Network Test RMSE: {test_rmse:.2f}")
      
      return test_rmse

