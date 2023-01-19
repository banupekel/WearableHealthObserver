"""Utility functions for training and dataset preparation"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from who.common import storage_path, dataset_config_file

def load_X(X_signals_paths):
    """Load "X" (the neural network's training and testing inputs)"""
    X_signals = []
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    """Load "y" (the neural network's training and testing outputs)"""
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

#  --------------------------------------------------------------------------------
def extract_batch_size(_train, step, batch_size):
    """Function to fetch a "batch_size" amount of data from "(X|y)_train" data."""
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]
    return batch_s

def one_hot(y_, n_classes):
    """Function to encode neural one-hot output labels from number indexes"""
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def save_to_file(file_name="results.txt", text="enivicivokki"):
    file_path = os.path.join(storage_path, file_name)
    with open(file_path, "a+") as file:
        file.write(text+"\n")
        file.close()

def get_input_signal_types():
    """returns signal types according to preferred dataset"""
    with open(dataset_config_file, "r") as file:
        #start reading from line 2
        next(file)
        for line in file:
            line = line.split(",")
            return line[:len(line)-1]

def get_labels():
    """returns dataset labels according to preferred dataset"""
    with open(dataset_config_file, "r") as file:
        for line in file:
            line = line.split(",")
            return line[:len(line)-1]

def loss_acc_plot(train_losses, test_losses, train_accuracies, test_accuracies, batch_size, display_iter, training_iters):
    """Function to plot accuracy and loss"""
    font = {
        'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 18
    }
    matplotlib.rc('font', **font)
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")
    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
        [training_iters]
    )
    plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")
    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')
    plt.show()

# -------------------------------------------------------------------------------
def confusion_matrix(one_hot_predictions, accuracy, n_classes, labels, y_test):
    """function to plot the multi-class confusion matrix and metrics """
    predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}%".format(100*accuracy))

    print("")
    print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
    print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # Plot Results:
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()