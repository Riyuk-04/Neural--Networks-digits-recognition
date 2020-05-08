import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from Network import Network

epochs = 30
network_architecture = [784,20,10]

train_images_file = 'train-images.idx3-ubyte'
train_images_array = idx2numpy.convert_from_file(train_images_file)
input_matrix = np.reshape(train_images_array,(60000,28*28))/500

train_labels_file = 'train-labels.idx1-ubyte'
train_labels_array = idx2numpy.convert_from_file(train_labels_file)

test_images_file = 't10k-images.idx3-ubyte'
test_images_array = idx2numpy.convert_from_file(test_images_file)
input_matrix_test = np.reshape(test_images_array,(10000,28*28))
input_matrix_test= np.reshape(test_images_array,(10000,28*28))/500

test_labels_file = 't10k-labels.idx1-ubyte'
test_labels_array = idx2numpy.convert_from_file(test_labels_file)

network1=Network(network_architecture)

network1.SGD(input_matrix,train_labels_array,epochs,input_matrix_test,test_labels_array)
