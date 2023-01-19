import numpy as np
import os
import tensorflow as tf
from who.data import  uci_har, pedometer, wisdm
from who.utils import(
 extract_batch_size,
 load_X,
 load_y,
 one_hot,
 loss_acc_plot,
 confusion_matrix,
 save_to_file,
 get_labels,
 get_input_signal_types  
)

from who.common import dataset_path, train, test
from who.model import LSTM_RNN

os.system("clear")
tf.compat.v1.disable_eager_execution()

# --------------------------------
# uncomment the dataset you want to train
wisdm()
# uci_har()
# pedometer()
# --------------------------------

# Define absolute paths for test and training datas
X_train_signals_paths = [os.path.join(dataset_path, train, "Inertial Signals", signal + "train.txt") for signal in get_input_signal_types()]
X_test_signals_paths = [os.path.join(dataset_path, test, "Inertial Signals", signal + "test.txt") for signal in get_input_signal_types()]
y_train_path = os.path.join(dataset_path, train, "y_train.txt")
y_test_path = os.path.join(dataset_path, test, "y_test.txt")

# Load dataset
X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)
y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

#   Additionnal Parameters --------------------------------------------------------------------------------
# Input Data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep
# LSTM Neural Network's internal structure
n_hidden = 32 # Hidden layer num of features
n_classes = len(get_labels()) # Total classes (should go up, or should go down)
# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 20  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training
out_file_name = "results.txt"
# Some debugging info
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


# Build the model
# build the neural network  --------------------------------------------------------------------------------
# Graph input/output
x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
# Graph weights
weights = {
    'hidden': tf.Variable(tf.random.normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random.normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random.normal([n_hidden])),
    'out': tf.Variable(tf.random.normal([n_classes]))
}
pred = LSTM_RNN(x, weights, biases,n_input=n_input, n_hidden=n_hidden, n_steps=n_steps)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# train the neural network  --------------------------------------------------------------------------------
# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(log_device_placement=True))
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(y_=extract_batch_size(y_train, step, batch_size), n_classes=n_classes)

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs,
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)

    # Evaluate network only at some steps for faster training:
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):

        # To not spam console, show training accuracy/loss in this "if"
        training_set_out = "Training iter #" + str(step*batch_size) + ":   Batch Loss = " + "{:.6f}".format(loss) + ", Accuracy = {}".format(acc)
        print(training_set_out)
        save_to_file(out_file_name, training_set_out)
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy],
            feed_dict={
                x: X_test,
                y: one_hot(y_=y_test, n_classes=n_classes)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        test_set_out = "PERFORMANCE ON TEST SET: " + "Batch Loss = {}".format(loss) + ", Accuracy = {}".format(acc)
        print(test_set_out)
        save_to_file(out_file_name, test_set_out)

    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_=y_test, n_classes=n_classes)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))


loss_acc_plot(train_losses, test_losses, train_accuracies, test_accuracies, batch_size, display_iter, training_iters)
confusion_matrix(one_hot_predictions, accuracy, n_classes, get_labels(), y_test)