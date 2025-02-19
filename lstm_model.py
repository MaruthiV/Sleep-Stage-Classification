import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self, input_shape, num_classes, learning_rate=0.0001, dropout_rate=0.2):
        self.input_shape = input_shape  # (time_steps, features)
        self.num_classes = num_classes  # Number of sleep stages
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def build_model(self):
        # Input layer
        input_layer = Input(shape=self.input_shape)

        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
        x = Dropout(self.dropout_rate)(x)

        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(self.dropout_rate)(x)

        # Fully connected layer
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)

        # Output layer
        output = Dense(self.num_classes, activation='softmax')(x)

        # Define the model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model

if __name__ == "__main__":
    input_shape = (300, 100)  # Example shape: 300 time steps, 100 features
    num_classes = 5  # Number of sleep stages: W, N1, N2, N3, REM

    lstm_model = LSTMModel(input_shape=input_shape, num_classes=num_classes)
    model = lstm_model.build_model()
    model.summary()
