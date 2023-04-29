import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
mnist = tf.keras.datasets.mnist # test dataset, can delete later
#https://www.tensorflow.org/tutorials/quickstart/beginner


# Load the dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Check if the dataset was loaded correctly
if x_train is not None and x_test is not None:
    print("Loaded data correctly")
else:
    print("Not loaded correctly")

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create model with 3 layers, dense in the end Dense(10) indicates the size of the output vector, this is a linear layer since there is no activation function specified 
model = tf.keras.models.Sequential([ 
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# In this example, our model returns a vector of logits/log-odd scores one for each class
predictions = model(x_train[:1]).numpy()
predictions
print(predictions) #REMOVE LATER, JUST TO TEST!!!!!!!!!!!!!!!!!!!

# tf.nn.softmax converts logits to probabilities for each class
tf.nn.softmax(predictions).numpy()
print(tf.nn.softmax(predictions).numpy()) #REMOVE LATER, JUST TO TEST!!!!!!!!!!!!

#Note: It is possible to bake the tf.nn.softmax function into the activation function for the last layer of the network. 
#While this can make the model output more directly interpretable, this approach is discouraged as it's impossible to provide an exact and numerically stable loss calculation for all models when using a softmax output.

# Define loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#  In this case, the loss function takes two inputs: a vector of the correct values for each example and a vector of the model's predicted values (logits) for each example.

loss_fn(y_train[:1], predictions).numpy()
print("loss_fn:", loss_fn(y_train[:1], predictions).numpy()) #REMOVE LATER, JUST TO TEST!!!!!!!!!!!!

# Configure and compile model with arguments and then we can train model
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Minimize loss during the training
model.fit(x_train, y_train, epochs=5)

# Checks models performance, typically on a validation set or a test set
model.evaluate(x_test,  y_test, verbose=2)
print(model.evaluate(x_test,  y_test, verbose=2)) #REMOVE LATER, JUST TO TEST!!!!!!!!!!!!

# To return probabilty of model, wrap the trained model from previous, and then attach softmax to it
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])



