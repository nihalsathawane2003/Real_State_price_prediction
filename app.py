from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset (ensure the path is correct for deployment)
dataset = pd.read_csv('HousingPrices-Amsterdam-August-2021.csv')

# Drop non-numeric columns or those not suitable for direct use in the model
columns_to_drop = ['Unnamed: 0', 'Address']
dataset = dataset.drop(columns=columns_to_drop)

# Handle categorical variables using one-hot encoding
dataset = pd.get_dummies(dataset, columns=['Zip'], drop_first=True)

# Handle missing values in the target variable
dataset['Price'] = dataset['Price'].fillna(dataset['Price'].mean())

# Identify important features
X = dataset.drop('Price', axis=1)
y = dataset['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a regression model (Random Forest Regressor in this example)
model = RandomForestRegressor()

# Fit the model
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        area = float(request.form['area'])
        rooms = int(request.form['rooms'])
        lon = float(request.form['lon'])
        lat = float(request.form['lat'])

        # Make sure the input_data length matches the feature columns
        input_data = [area, rooms, lon, lat] + [0] * (X_train.shape[1] - 4)

        # Reshape the input data to be a 2D array
        input_data_2d = [input_data]

        # Make the prediction
        prediction = model.predict(input_data_2d)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
