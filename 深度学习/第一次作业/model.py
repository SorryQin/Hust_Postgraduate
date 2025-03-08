# coding=utf-8
import numpy as np
import time
from utils import plot_confusion_matrix, plot_misclassified_images
from sklearn.metrics import confusion_matrix
from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer


class MNIST_MLP(object):
    def __init__(self, batch_size=30, input_size=784, hidden1=256, hidden2=128, hidden3=64, out_classes=10, lr=0.01,
                 max_epoch=30, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.train_losses = []

    def build_model(self):
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.hidden3)
        self.relu3 = ReLULayer()
        self.fc4 = FullyConnectedLayer(self.hidden3, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]

    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        self.fc4.load_param(params['w4'], params['b4'])

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        params['w4'], params['b4'] = self.fc4.save_param()
        np.save(param_dir, params)

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        h3 = self.relu3.forward(h3)
        h4 = self.fc4.forward(h3)
        prob = self.softmax.forward(h4)
        return prob

    def backward(self):
        dloss = self.softmax.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.relu3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self, train_data):
        max_batch = train_data.shape[0] // self.batch_size
        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            np.random.shuffle(train_data)
            for idx_batch in range(max_batch):
                batch_images = train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.train_losses.append(loss)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def evaluate(self, test_data):
        pred_results = np.zeros([test_data.shape[0]])
        num_batches = test_data.shape[0] // self.batch_size
        for idx in range(num_batches):
            batch_images = test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            start = time.time()
            prob = self.forward(batch_images)
            end = time.time()
            print("inferencing time: %f" % (end - start))
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
        # 处理剩余数据
        remaining_data = test_data[num_batches * self.batch_size:, :-1]
        if remaining_data.shape[0] > 0:
            prob = self.forward(remaining_data)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[num_batches * self.batch_size:] = pred_labels
        accuracy = np.mean(pred_results == test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)

        # 计算混淆矩阵
        cm = confusion_matrix(test_data[:, -1], pred_results)
        plot_confusion_matrix(cm, classes=range(self.out_classes))

        # 可视化分类错误案例
        misclassified_indices = np.where(pred_results != test_data[:, -1])[0]
        plot_misclassified_images(test_data[misclassified_indices, :-1],
                                  test_data[misclassified_indices, -1],
                                  pred_results[misclassified_indices])