from tensorflow.keras.models import load_model

model = load_model('Example Model/deep_neural_network_model', compile = True)

features = [[108, 160, 341, 181, 21, 2, 21, 49, 22]]

prediction = model.predict(features)
print(prediction)