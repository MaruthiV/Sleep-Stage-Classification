import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, GlobalMaxPooling1D, TimeDistributed
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import CRF

class CNNCRFModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape  # (time_steps, features)
        self.num_classes = num_classes  # Number of sleep stages

    def build_model(self):
        # Input layer
        input_layer = Input(shape=self.input_shape)

        # Time-distributed CNN layers
        x = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='tanh'))(input_layer)
        x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
        x = TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'))(x)
        x = TimeDistributed(GlobalMaxPooling1D())(x)

        # Bi-directional LSTM layer
        x = Bidirectional(LSTM(64, return_sequences=True))(x)

        # Dense layer
        x = TimeDistributed(Dense(64, activation='relu'))(x)
        x = Dropout(0.3)(x)

        # CRF layer
        crf = CRF(self.num_classes)
        output = crf(x)

        # Define the model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=crf.loss, metrics=[crf.accuracy])

        return model

if __name__ == "__main__":
    input_shape = (500, 200)  
    num_classes = 5 

    cnn_crf = CNNCRFModel(input_shape=input_shape, num_classes=num_classes)
    model = cnn_crf.build_model()
    model.summary()
