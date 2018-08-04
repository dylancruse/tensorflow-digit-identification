import tensorflow as tf
from define_model import x, y, y_, train_step

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Launch the model in an interactive session
sess = tf.InteractiveSession()

#Initialize Variables
tf.global_variables_initializer().run()

#Train the model in batches of 100
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""
Testing for accuracy
"""
predicted_label = tf.argmax(y, 1)
correct_label = tf.argmax(y_, 1)

#List of booleans where the prediction was correct
correct_predictions = tf.equal(predicted_label, correct_label)

#Cast to floats and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

#Run the test data, check accuracy
acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Model Accuracy: %{:.2f}".format(acc * 100))