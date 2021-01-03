import numpy as np
import pandas as pd
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import load_model

class user():
    def __init__(self, name):
        with open(f'user_data/{name}.csv', mode='w') as myfile:
            writer = csv.writer(myfile)
            writer.writerow(['Meal', 'Bedtime', 'Waketime', 'Quality', 'Electronics', 'Up', 'Temperature', 'Noise', 'Nap'])
        
        self.datafile = name + '.csv'
        self.deep_neural_network_model = None
        self.variables = ['Mealtime', 'Bedtime', 'Waketime', 'Quality', 'Electronics', 'Up', 'Temperature', 'Noise', 'Nap']
        self.variables_improvement = [-60, -240, 240, None, -60, -5, -5, -25, -25]
        self.user_messages = ['Try to have your meal % minutes earlier', 'Try to go to sleep % minutes earlier',
                              'Try to wake up % minutes later', None, 'Try to stop using electronics % minutes earlier',
                              'Try to lower the temperature by % degrees Celsius', 'Try to find a way to lower the noise level by % decibels',
                              'Try to take % minutes less of naps']

    def make_prediction(self, features):
        """Given current features, make a prediction"""
        model = load_model(self.deep_neural_network_model, compile = True)
        prediction = model.predict(features)

        if prediction > 10:
            return 10
        else:
            return prediction

    def message(self, features, prediction):
        """Given a prediction, display a message to the user"""
        if prediction >= 0 and prediction < 2.5:
            change_multiplier = 1
        elif prediction >= 2.5 and prediction < 5:
            change_multiplier = 0.5
        elif prediction >= 5 and prediction < 7.5:
            change_multiplier = 0.25
        else:
            change_multiplier = 0.125

        self.check_all_hypothetical(features, change_multiplier , prediction)

    def make_hypothetical_prediction(self, features, feature_ind, change):
        """Given current features, make a prediction with features adjusted"""
        hypothetical_features = features.copy()
        hypothetical_features[feature_ind] += change
        return self.make_prediction(hypothetical_features)

    def check_all_hypothetical(self, features, change_mult, prediction):
        solutions = ""
        for i in range(len(self.variables)):
            if i != self.variables.index('Quality'):
                hyp_prediction = self.make_hypothetical_prediction(features, i, self.variables_improvement[i] * change_mult)
                if hyp_prediction >= prediction:
                    solutions += (self.user_messages[i].replace('%', str(self.variables_improvement[i] * change_mult))
                                  + f"your quality score may increase to {hyp_prediction}! \n")
        return solutions

    def train(self):
        """Trains the deep neural network model with data from self.datafile"""
        raw_dataset = pd.read_csv(self.datafile, sep = ',', header = 0,
                                  na_values = '?', comment = '\t',
                                  skipinitialspace = True)

        dataset = raw_dataset.copy()
        dataset.tail()

        # Clear unknown values
        dataset.isna().sum()
        dataset = dataset.dropna()

        # takes a sample of 80% of the data points
        train_dataset = dataset.sample(frac = 0.8, random_state = 0)
        test_dataset = dataset.drop(train_dataset.index)

        # Split features from labels for training and test datasets
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()
        train_labels = train_features.pop('Quality')
        test_labels = test_features.pop('Quality')

        # normalize data
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(train_features))

        # builds the model
        def build_and_compile_model(norm):
          model = keras.Sequential([
              norm,
              layers.Dense(64, activation='relu'),
              layers.Dense(64, activation='relu'),
              layers.Dense(1)
          ])

          model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(0.001))
          return model

        deep_neural_network_model = build_and_compile_model(normalizer)

        history = deep_neural_network_model.fit(
            train_features, train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)

        deep_neural_network_model.save('user_data/deep_neural_network_model')
        self.deep_neural_network_model = 'user_data/deep_neural_network_model'
