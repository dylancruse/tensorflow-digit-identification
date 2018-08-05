import tensorflow as tf

"""
Create placeholder for images.
In the form of a 2D tensor of floats with shape [None, 784].
None allows the dimension to be any length.
Each image is 28x28 pixels, flattened into a 784-dimensional vector.
"""

image_vectors = tf.placeholder(tf.float32, [None, 784])

"""
Create Variables for weights and biases of the model.
Variables are modifiable tensors used in tf's operations.
"""

weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

"""
Define the model.
-Multiply x by W with .matmul
-Add b with vector addition
-Apply softmax regression
"""

smax_model = tf.nn.softmax(tf.matmul(image_vectors, weights) + biases)
