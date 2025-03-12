import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

MNIST_DIR = "E:/研一下/deepl/MNIST/mnist"
TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL = "train-labels.idx1-ubyte"
TEST_DATA = "t10k-images.idx3-ubyte"
TEST_LABEL = "t10k-labels.idx1-ubyte"

def load_mnist(file_dir, is_images=True):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data

def load_data():
    print('Loading MNIST data from files...')
    train_images = load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
    train_labels = load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
    test_images = load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
    test_labels = load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
    train_data = np.append(train_images, train_labels, axis=1)
    test_data = np.append(test_images, test_labels, axis=1)
    return train_data, test_data

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

def plot_misclassified_images(images, true_labels, pred_labels, num_images=9):
    indices = np.random.choice(len(images), num_images)
    plt.figure(figsize=(3, 3))
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {int(true_labels[idx])}, Pred: {int(pred_labels[idx])}')
        plt.axis('off')
    plt.show()