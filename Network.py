import numpy as np
import idx2numpy
import random

learning_rate = 0.1

def sigmoid(z):
	a = 1.0/(1+np.exp(-z))
	return a


class Network(object):
	def __init__(self,structure):
		self.length = len(structure)
		self.structure = structure
		self.biases = [np.random.randn(1,y) for y in structure[1:]]
		self.weights = [np.random.randn(y,x) for (y,x) in zip(structure[1:],structure[:-1])]

	def forward_prop1(self,input,layer):
		a = sigmoid(np.dot(self.weights[layer-1],input)+self.biases[layer-1])
		return a

	def forward_prop(self,input):
		a = sigmoid(np.dot(self.weights,input)+self.biases)
		return a

	def output(self,input_matrix,count_example):
		count_layer = 1 
		output_array = input_matrix[count_example - 1]

		while(count_layer<self.length):
			output_array = np.squeeze(self.forward_prop1(output_array,count_layer))	
			count_layer += 1 

		return output_array

	def SGD(self,input_matrix,train_labels_array,epochs,input_matrix_test,test_labels_array):
		for x in range(epochs):
			self.update_weights(input_matrix,train_labels_array)

			count = 0
			success = 0
			while(count < np.shape(input_matrix_test)[0]):
				if (np.argmax(self.output(input_matrix_test,count+1)) == test_labels_array[count]):
					success +=1
				count += 1

			print("Epoch completed:",x+1)
			print("success = ",success,"/",np.shape(input_matrix_test)[0],"\n")

	def update_weights(self,input_matrix,train_labels_array):
		count = 0
		success = 0;
		while(count < np.shape(input_matrix)[0]):
			output_train = np.zeros(10)
			output_train[train_labels_array[count]] = 1
			DELTA_weights,DELTA_biases = self.back_prop(input_matrix[count],output_train)

			self.weights = [w - learning_rate*Delw for w,Delw in zip(self.weights,DELTA_weights)]
			self.biases = [b - learning_rate*np.transpose(Delb) for b,Delb in zip (self.biases,DELTA_biases)]
			count += 1	

	def back_prop(self,input_array,label):
		DELTA_weights = [np.zeros(w.shape) for w in self.weights]
		DELTA_biases = [np.zeros(b.shape) for b in self.biases]
		layer_error = [np.zeros(x) for  x in self.structure[1:]]

		activation = input_array
		activations = [input_array]

		#forward-propagation

		for w,b in zip(self.weights,self.biases):
			activation = np.squeeze(sigmoid(np.dot(w,activation)+b))
			activations.append(activation)

		#back-propogation	

		delta = activations[-1] -label
		delta = delta.reshape(len(delta),1)
		activations[-2] = activations[-2].reshape(len(activations[-2]),1)
		DELTA_biases[-1] = delta
		DELTA_weights[-1] = np.dot(delta,np.transpose(activations[-2])) 

		for x in range(2,self.length):
			acti_derivative = np.multiply(activations[-x],1-activations[-x])
			delta = np.multiply(np.dot(np.transpose(self.weights[-x+1]),delta),acti_derivative)
			activations[-x-1] = activations[-x-1].reshape(len(activations[-x-1]),1)
			DELTA_biases[-x] = delta
			DELTA_weights[-x] = np.dot(delta,np.transpose(activations[-x-1]))

		return (DELTA_weights,DELTA_biases)

		




























		

		
			





		





