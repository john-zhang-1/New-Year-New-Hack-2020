import numpy as np
import pandas as pd
import csv
from os import path
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import load_model

import IPython
import kerastuner as kt

class user():
    def __init__(self, name):
        # set the paths for user's datafile and model
        self.folder = f'user_data/{name}'
        self.datafile = f'{self.folder}/{name}_data.csv'
        self.deep_neural_network_model = f'{self.folder}/{name}_model'

        # common attributes to each user
        self.variables = ['Mealtime', 'Bedtime', 'Waketime', 'Quality', 'Electronics', 'Up', 'Temperature', 'Noise', 'Nap']
        self.features = self.variables.copy()
        self.label = self.features.pop(self.features.index('Quality'))
        self.features_improvement = [-60, -240, 240, 60, -5, -5, -25, -25]
        self.user_messages = ['Try to have your meal % minutes earlier, ',
                              'Try to go to sleep % minutes earlier, ',
                              'Try to wake up % minutes later, ',
                              'Try to stop using electronics % minutes earlier, ',
                              'Try to wake up % times less, ',
                              'Try to lower the temperature by % degrees Celsius, ',
                              'Try to find a way to lower the noise level by % decibels, ',
                              'Try to take % minutes less of naps, ']

        # creates a folder called called user_data for the first user
        Path(self.folder).mkdir(parents=True, exist_ok=True)

        # creates a datafile for a user if they don't have one yet
        if not path.exists(self.datafile):
            with open(self.datafile, mode='w', newline='') as my_file:
                writer = csv.writer(my_file)
                writer.writerow(self.variables)

    def input_data(self, data):
        """Inputs given data into the data document"""

        # checks for valid number of variables
        if len(data) != len(self.variables):
            print('Invalid data')
            return

        # adds to the end of the datafile
        with open(self.datafile, mode='a', newline='') as my_file:
            writer = csv.writer(my_file)
            writer.writerow(data)

    def make_prediction(self, features):
        """Given current features, make a prediction"""
        model = load_model(self.deep_neural_network_model, compile = True)
        prediction = model.predict(features)

        # prediction cannot be higher than 10
        if prediction[0][0] > 10:
            return 10
        else:
            return prediction[0][0]

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

        return self.check_all_hypothetical(features, change_multiplier, prediction)

    def make_hypothetical_prediction(self, features, feature_ind, change):
        """Given current features, make a prediction with features adjusted"""
        hypothetical_features = features.copy()

        hypothetical_features[0][feature_ind] += change

        return self.make_prediction(hypothetical_features)

    def check_all_hypothetical(self, features, change_mult, prediction):
        """Checks all hypothetical predictions based on the change_mult multiplied by the set improvements"""
        solutions = ""
        for i in range(len(self.features)):
            change = self.features_improvement[i] * change_mult
            hyp_prediction = self.make_hypothetical_prediction(features, i, change)
            if hyp_prediction > prediction:
                solutions += (self.user_messages[i].replace('%', str(change))
                              + f"your quality score may increase to {hyp_prediction}! \n")
        return solutions

    def copy_data(self, old_datafile):
        """Copies data from a csv file to self.datafile"""
        with open(self.datafile, mode='w', newline='') as f2:
            writer = csv.writer(f2)
            with open(old_datafile, newline='') as infile:
                reader = csv.reader(infile)
                for row in reader:
                    writer.writerow(row)

    def convert_time_to_minutes(self, time):
        """Takes in a time in the form of hh:mm and converts to minutes"""
        return int(time[:2]) * 60 + int(time[3:])

    def train(self):
        """Trains the deep neural network model with data from self.datafile"""
        raw_dataset = pd.read_csv(self.datafile, sep = ',', header = 0,
                                  na_values = '?', comment = '\t', skipinitialspace = True)

        dataset = raw_dataset.copy()
        dataset.tail()

        # Clear unknown values
        dataset.isna().sum()
        dataset = dataset.dropna()

        # takes a sample of 90% of the data points
        train_dataset = dataset.sample(frac = 0.9, random_state = 0)
        test_dataset = dataset.drop(train_dataset.index)

        # Split features from labels for training and test datasets
        train_x = train_dataset.copy()
        test_x = test_dataset.copy()
        train_y = train_x.pop('Quality')
        test_y = test_x.pop('Quality')

        # normalize data
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(train_x))

        def build_and_compile_model(norm):
          model = keras.Sequential([
              norm,
              layers.Dense(64, activation='relu'),
              layers.Dense(64, activation='relu'),
              layers.Dense(1)
          ])

          model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
          return model

        model = build_and_compile_model(normalizer)
        model.fit(train_x, train_y, validation_split=0.2, verbose=0, epochs=100)
        print('Mean absolute error of dnn_model: ' + str(model.evaluate(test_x, test_y, verbose = 0)))

        model.save(str(self.deep_neural_network_model))

    def build_model(self, hyperparameters):
        model = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=["max"])
        return model

    def instantiate_tuning(self):

        tuner = kt.Hyperband(self.build_model,
                             objective = kt.Objective('Quality', 'max'),
                             max_epochs=30,
                             factor=3,
                             directory= f'{self.folder}/John_dir',
                             project_name= 'intro_to_kt')

        class ClearTrainingOutput(tf.keras.callbacks.Callback):
            def on_train_end(*args, **kwargs):
                IPython.display.clear_output(wait=True)

        raw_dataset = pd.read_csv(self.datafile, sep=',', header=0,
                                  na_values='?', comment='\t', skipinitialspace=True)

        dataset = raw_dataset.copy()
        dataset.tail()

        # Clear unknown values
        dataset.isna().sum()
        dataset = dataset.dropna()

        # takes a sample of 90% of the data points
        train_dataset = dataset.sample(frac=0.9, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        # Split features from labels for training and test datasets
        train_x = train_dataset.copy()
        test_x = test_dataset.copy()
        train_y = train_x.pop('Quality')
        test_y = test_x.pop('Quality')

        tuner.search(train_x, train_y, epochs=10, validation_data=(test_x, test_y),
                     callbacks=[ClearTrainingOutput()])

        # Get the optimal hyper-parameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""Optimal number of units in the first layer: {best_hps.get('units')}\n
        Optimal learning rate: {best_hps.get('learning_rate')}.
        """)


if __name__ == '__main__':
    pass