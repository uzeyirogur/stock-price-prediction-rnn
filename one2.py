import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import os

# Function to load and preprocess the dataset
def load_and_preprocess_data(train_path, test_path):
    # Load training set
    dataset_train = pd.read_csv(train_path)
    training_set = dataset_train.iloc[:, 1:2].values
    
    # Feature scaling
    scaler = MinMaxScaler()
    training_set_scaled = scaler.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and 1 output
    x_train, y_train = [], []
    for i in range(60, len(training_set_scaled)):
        x_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Reshaping for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Load test set
    dataset_test = pd.read_csv(test_path)
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    # Combining train and test for prediction input
    dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    # Creating test set structure
    x_test = []
    for i in range(60, len(inputs)):
        x_test.append(inputs[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_train, y_train, x_test, real_stock_price, scaler

# Function to build the RNN model
def build_rnn(input_shape):
    regressor = Sequential()
    
    # Adding LSTM layers with Dropout
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    
    # Output layer
    regressor.add(Dense(units=1))
    
    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    return regressor

# Function to visualize the results
def visualize_results(real_price, predicted_price):
    plt.plot(real_price, color="red", label="Real Google Stock Price")
    plt.plot(predicted_price, color="blue", label="Predicted Google Stock Price")
    plt.title("Google Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Google Stock Price")
    plt.legend()
    plt.show()

# Function to save the model
def save_model(model, filename="stock_price_rnn.h5"):
    model.save(filename)
    print(f"Model saved to {filename}")

# Function to load the model
def load_saved_model(filename="stock_price_rnn.h5"):
    if os.path.exists(filename):
        print(f"Loading model from {filename}")
        return load_model(filename)
    else:
        print(f"Model file {filename} not found. Please train the model first.")
        return None

# Main function
def main():
    # Paths to the datasets
    train_path = "data/Google_Stock_Price_Train.csv"
    test_path = "data/Google_Stock_Price_Test.csv"
    
    # Preprocessing data
    x_train, y_train, x_test, real_stock_price, scaler = load_and_preprocess_data(train_path, test_path)
    
    # Check if model already exists, if not train it
    model_filename = "stock_price_rnn.h5"
    regressor = load_saved_model(model_filename)
    
    if regressor is None:
        # If no saved model is found, train a new model
        regressor = build_rnn(input_shape=(x_train.shape[1], 1))
        regressor.fit(x_train, y_train, epochs=100, batch_size=32)
        save_model(regressor, model_filename)
    
    # Making predictions
    predicted_stock_price = regressor.predict(x_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    # Visualizing the results
    visualize_results(real_stock_price, predicted_stock_price)

# Run the main function
if __name__ == "__main__":
    main()
