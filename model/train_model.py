import tensorflow as tf
from model.define_model import image_vectors, smax_model

"""
Define training.
Create placeholder for correct answers (one-hot vector with digit labels).
Implement cross entropy to determine loss of the model.
Uses backpropagation and gradient descent to tweak Variables to reduce loss. 
"""

correct_labels = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(
    correct_labels * tf.log(smax_model), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
