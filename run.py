import tensorflow as tf
from model.define_model import image_vectors, smax_model
from model.train_model import correct_labels, train_step

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Launch the model in an interactive session
sess = tf.InteractiveSession()

#Initialize Variables
tf.global_variables_initializer().run()

#Train the model in batches of 100
for _ in range(1000):
    batch_images, batch_labels = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={
        image_vectors: batch_images, 
        correct_labels: batch_labels
    })

"""
Testing for accuracy
"""
predicted_label = tf.argmax(smax_model, 1)
accurate_label = tf.argmax(correct_labels, 1)

#List of booleans where predicted label matched correct label
correct_predictions = tf.equal(predicted_label, accurate_label)

#Cast to floats and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

#Run the test data, check accuracy
acc = sess.run(accuracy, feed_dict={
    image_vectors: mnist.test.images, 
    correct_labels: mnist.test.labels
})

print("Model Accuracy: %{:.2f}".format(acc * 100))