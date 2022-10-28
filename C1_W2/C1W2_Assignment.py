#!/usr/bin/env python
# coding: utf-8

# # Week 2: Implementing Callbacks in TensorFlow using the MNIST Dataset
# 
# In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.
# 
# Write an MNIST classifier that trains to 99% accuracy and stops once this threshold is achieved. In the lecture you saw how this was done for the loss but here you will be using accuracy instead.
# 
# Some notes:
# 1. Your network should succeed in less than 9 epochs.
# 2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!" and stop training.
# 3. If you add any additional variables, make sure you use the same names as the ones used in the class. This is important for the function signatures (the parameters and names) of the callbacks.

# In[ ]:


import os
import tensorflow as tf
from tensorflow import keras

current_dir = os.getcwd()


data_path = os.path.join(current_dir, "data/mnist.npz")


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)
        

x_train = x_train / 255.0


data_shape = x_train.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

class myCallback():
        # Define the correct function signature for on_epoch_end
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') is not None and logs.get('accuracy') > 0.99):
                print("\nReached 99% accuracy so cancelling training!") 
                
                # Stop training once the above condition is met
                self.model.stop_training = True

### END CODE HERE


# GRADED FUNCTION: train_mnist
def train_mnist(x_train, y_train):

    ### START CODE HERE
    
    # Instantiate the callback class
    callbacks = None
    
    # Define the model
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Flatten(input_shape(28,28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]) 

    # Compile the model
    model.compile(optimizer='tf.optimizers.Adam()', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
    
    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    ### END CODE HERE

    return history


# Call the `train_mnist` passing in the appropiate parameters to get the training history:

# In[16]:


hist = train_mnist(x_train, y_train)


# If you see the message `Reached 99% accuracy so cancelling training!` printed out after less than 9 epochs it means your callback worked as expected. 

# ## Need more help?
# 
# Run the following cell to see an architecture that works well for the problem at hand:

# In[ ]:


# WE STRONGLY RECOMMEND YOU TO TRY YOUR OWN ARCHITECTURES FIRST
# AND ONLY RUN THIS CELL IF YOU WISH TO SEE AN ANSWER

# import base64

# encoded_answer = "CiAgIC0gQSBGbGF0dGVuIGxheWVyIHRoYXQgcmVjZWl2ZXMgaW5wdXRzIHdpdGggdGhlIHNhbWUgc2hhcGUgYXMgdGhlIGltYWdlcwogICAtIEEgRGVuc2UgbGF5ZXIgd2l0aCA1MTIgdW5pdHMgYW5kIFJlTFUgYWN0aXZhdGlvbiBmdW5jdGlvbgogICAtIEEgRGVuc2UgbGF5ZXIgd2l0aCAxMCB1bml0cyBhbmQgc29mdG1heCBhY3RpdmF0aW9uIGZ1bmN0aW9uCg=="
# encoded_answer = encoded_answer.encode('ascii')
# answer = base64.b64decode(encoded_answer)
# answer = answer.decode('ascii')

# print(answer)


# **Congratulations on finishing this week's assignment!**
# 
# You have successfully implemented a callback that gives you more control over the training loop for your model. Nice job!
# 
# **Keep it up!**
