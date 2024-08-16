import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset
dataset = pd.read_csv('C:\\Users\\Nihal\\Desktop\\land\\HousingPrices-Amsterdam-August-2021.csv')

# Convert 'Price' column to numeric, ignoring errors
dataset['Price'] = pd.to_numeric(dataset['Price'], errors='coerce')

# Impute missing values with the mean on the entire dataset
dataset['Price'].fillna(dataset['Price'].mean(), inplace=True)

#print(dataset.isnull().sum())

# Drop non-numeric columns or those not suitable for direct use in the model
columns_to_drop = ['Unnamed: 0', 'Address']
dataset = dataset.drop(columns=columns_to_drop)

# Handle categorical variables using one-hot encoding
dataset = pd.get_dummies(dataset, columns=['Zip'], drop_first=True)

# Identify important features
X = dataset.drop('Price', axis=1)
y = dataset['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a regression model (Random Forest Regressor in this example)
model = RandomForestRegressor()

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize predicted vs actual values with informative labels
plt.scatter(y_test, y_pred, label="Predicted vs Actual")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()
