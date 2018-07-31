import tensorflow as tf

"""
Create placeholder for images.
In the form of a 2D tensor of floats with shape [None, 784].
None allows the dimension to be any length.
Each image is 28x28 pixels, flattened into a 784-dimensional vector.
"""

x = tf.placeholder(tf.float32, [None, 784])

"""
Create Variables for weights and biases of the model.
Variables are modifiable tensors used in tf's operations.
"""

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
Define the model.
-Multiply x by W with .matmul
-Add b with vector addition
-Apply softmax regression
"""

y = tf.nn.softmax(tf.matmul(x, W) + b)

"""
Define training functions.
Create placeholder for correct answers (one-hot vector with digit labels).
Implement cross entropy to determine loss of the model.
Uses backpropagation and gradient descent to tweak Variables to reduce loss. 
"""

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(
    y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
