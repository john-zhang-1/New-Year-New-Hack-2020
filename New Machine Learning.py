import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# column_names = ['Quality', 'Meal', 'Bedtime', 'Waketime', 'Length',	'Electronics', 'Up', 'Temperature',	'Noise', 'Nap']
raw_dataset = pd.read_csv('random_gen_data.csv', sep = ',', header = 0,
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

sns.pairplot(train_dataset[['Quality', 'Meal', 'Bedtime', 'Waketime', 'Length']], diag_kind = 'kde')
plt.show()

train_dataset.describe().transpose()

# Split features from labels

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Quality')
test_labels = test_features.pop('Quality')

# view variation
print(train_dataset.describe().transpose()[['mean', 'std']])


# normalize data

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision = 3, suppress = True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())


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
deep_neural_network_model.summary()

history = deep_neural_network_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

test_results = {}
test_results['deep_neural_network_model'] = deep_neural_network_model.evaluate(test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index = ['Mean absolute error [Quality]']).T)