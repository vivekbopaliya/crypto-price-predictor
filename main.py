import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

client_key = os.getenv("CLIENT_KEY")
client_secret = os.getenv("CLIENT_SECRET")

class CryptoPricePredictor:
    def __init__(self):
        self.client = Client(client_key, 
                             client_secret)
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 60  # Look at last 60 timestamps
        
    def fetch_historical_data(self, symbol='BTCUSDT', interval='1h', limit=1000):
        """Fetch historical klines/candlestick data from Binance."""
        klines = self.client.get_historical_klines(
            symbol, interval, f"{limit} hours ago UTC"
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignored'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.astype(float)
        return df

    def prepare_data(self, df, target_column='close'):
        """Prepare data for LSTM model."""
        data = df[target_column].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            # 1 if price went up, 0 if down
            y.append(1 if scaled_data[i] > scaled_data[i-1] else 0)
            
        return np.array(X), np.array(y)

    def build_model(self):
        """Create LSTM model architecture."""
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', 
                          metrics=['accuracy'])
        return self.model

    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model."""
        return self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def predict(self, X):
        """Generate predictions with probabilities."""
        return self.model.predict(X)

    def backtest(self, df, target_column='close', test_size=0.2):
        """Perform backtesting on historical data."""
        X, y = self.prepare_data(df, target_column)
        
        # Split data into training and testing sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        self.build_model()
        history = self.train(X_train, y_train)
        
        # Generate predictions
        predictions_prob = self.predict(X_test)
        predictions = (predictions_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions_prob,
            'actual': y_test,
            'history': history
        }

    def plot_results(self, results):
        """Visualize backtesting results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot training history
        ax1.plot(results['history'].history['accuracy'], label='Training Accuracy')
        ax1.plot(results['history'].history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot predictions vs actual
        ax2.plot(results['predictions'], label='Prediction Probability')
        ax2.plot(results['actual'], label='Actual')
        ax2.set_title('Predictions vs Actual')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        
        plt.tight_layout()
        return fig

# Example usage
def main():
    # Initialize predictor
    predictor = CryptoPricePredictor()
    
    # Fetch live data from Binance
    historical_data = predictor.fetch_historical_data(
        symbol='BTCUSDT',
        interval='1h',
        limit=1000
    )
    
    # Perform backtesting
    results = predictor.backtest(historical_data)
    
    print("\n🔍 **Backtesting Results:**")
    print(f"✅ **Accuracy:** {results['accuracy']:.2%}")
    print("\n📊 **Detailed Classification Report:**")
    print(results['classification_report'])
    
    # Plot results
    predictor.plot_results(results)
    plt.show()

if __name__ == "__main__":
    main()
