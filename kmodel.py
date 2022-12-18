from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np


# Class KModel

class KModel():
    def __init__(self, epochs_number, batchsize, number_layers, units, units_layer, X_train, y_train, X_test, y_test, n_column):
        self.epochs_number = epochs_number
        self.batchsize = batchsize
        self.number_layers = number_layers
        self.units = units
        self.units_layer = units_layer
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_column = n_column

    def generate_model(self):
        model = Sequential()
        model.add(Dense(self.units, activation='relu', input_shape=(self.n_column,)))
        for i in range(self.number_layers):
            model.add(Dense(self.units, activation='relu'))
            i += 1
        model.add(Dense(1, activation='sigmoid'))

        # Compile Model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit_evaluate_model(self):
        new_model = self.generate_model()
        new_model.fit(self.X_train, self.y_train, epochs=self.epochs_number, batch_size=self.batchsize)
        predictions = new_model.predict(self.X_test)

        # Evaluate the Model
        prediction_test = [round(pred[0]) for pred in predictions]

        evaluate_df = pd.DataFrame({'observed': self.y_test, 'predicted': prediction_test})
        evaluate_df['check'] = evaluate_df['observed'] - evaluate_df['predicted']
        accuracy = len(evaluate_df[evaluate_df['check']==0]) / len(evaluate_df)

        # print(f"Model with {self.number_layers} layers resulted an accuracy of {accuracy}")
        return accuracy
