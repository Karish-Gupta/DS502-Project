import pandas as pd
from sklearn.model_selection import train_test_split
from nn_model import PricePredictionNN

def preprocess(data: pd.DataFrame):
   # Levy has many missing values
   data['Levy'] = pd.to_numeric(data['Levy'], errors='coerce')
   levy_mean_val = data['Levy'].mean()
   data['Levy'].fillna(levy_mean_val, inplace=True)
   
   # Mileage
   data['Mileage'] = pd.to_numeric((data['Mileage']).astype(str).str.replace(' km', '', regex=True), errors='coerce')

   # Doors
   data['Doors'] = data['Doors'].astype(str).str.extract(r'(\d+)', expand=False).astype(int).map({4: 4, 2: 2, 5: 6})  # Map values

   # Engine Volume
   data['Engine volume'] = pd.to_numeric(data['Engine volume'].astype(str).str.replace(' Trubo', '', regex=True), errors='coerce')

   # Find age of car rather than having year of production
   data['Age'] = 2025 - data['Prod. year']
   data = data.drop(columns=['Prod. year'])

   # Adjust column names and remove spaces
   data.columns = data.columns.str.strip().str.replace(' ', '_')

   # Drop ID column 
   data = data.drop(columns=['ID'])
   data = data.dropna()

   # Handle outliers
   # Remove entries with extreme Prices
   Q1 = data['Price'].quantile(0.25)
   Q3 = data['Price'].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR

   data = data[(data['Price'] >= lower_bound) & (data['Price'] <= upper_bound)]

   return data

def split_data(data: pd.DataFrame):
   # Split data 
   train, test = train_test_split(data, test_size=0.2, random_state=123)
   y_train = train.pop('Price') # Keep only price column in y 
   X_train = train
   y_test = test.pop('Price')
   X_test = test

   return X_train, y_train, X_test, y_test

def OHE_data(data: pd.DataFrame):
   # OHE Categorical Data
   # Removed Model (This adds over 1500 more columns)
   data = data.drop(columns=['Model'])
   return pd.get_dummies(data, columns=['Manufacturer', 'Category', 'Leather_interior', 'Fuel_type', 'Gear_box_type', 'Drive_wheels', 'Wheel', 'Color'])


# Hyperparameter tuning function
def hyperparameter_tuning(lr_values, batch_size_values, epochs_values, decay_values, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
   best_rmse = float('inf')  # Start with a large value for best RMSE
   best_params = None
   best_model = None
     
   # Iterate through all combinations of hyperparameters
   for decay in decay_values:
      for lr in lr_values:
         for batch_size in batch_size_values:
               for epochs in epochs_values:
                  print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}, decay={decay}...")
                  
                  # Initialize the model
                  model = PricePredictionNN(input_dim=X_train_tensor.shape[1])
                  
                  # Train the model
                  model.train_model(X_train_tensor, y_train_tensor, epochs=epochs, lr=lr, batch_size=batch_size, decay=decay)
                  
                  # Evaluate the model and get test RMSE
                  print("Evaluating model...")
                  test_rmse = model.eval_model(X_test_tensor, y_test_tensor)
                                 
                  # Track the best RMSE and associated hyperparameters
                  if test_rmse < best_rmse:
                     best_rmse = test_rmse
                     best_params = (lr, batch_size, epochs, decay)
                     best_model = model
   
   return best_model, best_params, best_rmse