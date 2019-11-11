import numpy as  np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#input vectors

inputs = np.array ([[30, 40, 50],
                    [40, 50, 20],
                    [50, 20, 15],
                    [20, 15, 60],
                    [15, 60, 70],
                    [60, 70, 50] ])

#output vectors

outputs = np.array([ [20], [15], [60], [70], [50], [40] ])

scalar = MinMaxScaler()
scalar.fit(inputs)
#inputs being normalized
Norm_inputs = scalar.transform(inputs)

print(Norm_inputs, "\n")

scalar = MinMaxScaler()
scalar.fit(outputs)
#outputs being normalized
Norm_outputs = scalar.transform(outputs)

print(Norm_outputs, "\n")

#learning rate
learning_rate = 0.001

class NeuralNet:
    #variables
    def __init__(self, Norm_inputs, Norm_outputs, learning_rate):
        self.Norm_inputs = Norm_inputs
        self.Norm_outputs = Norm_outputs
        self.learning_rate = learning_rate

        #weights
        self.weights = np.array([ [.50], [.50], [.50] ])
        self.weights1 = np.array([[0.2, 0.3, 0.2], [0.1, 0.1, 0.1]])
        self.weights2 = np.array([[0.5, 0.1]])
        self.error_history = []
        self.epoch_list = []

    # calculate the sigmoid
    def sigmoid(self, x, deriv = False):
        if deriv == True:
            return x * (1-x)
        return 1 / (1 + np.exp(-x))

    #data will flow through the neural network
    def feed_forward(self):
        layer1 = self.sigmoid(np.dot(self.weights1, self.Norm_inputs))
        layer2 = self.sigmoid(np.dot(self.weights2, layer1))
        self.hidden = layer2
        return self.hidden

    def backpropagation(self):
        hidden = self.feed_forward()

        self.error = self.Norm_outputs - hidden
        delta = self.error * self.sigmoid(hidden, deriv=True)

        output_vector1 = np.dot(self.weights1, Norm_inputs)
        output_vector_hidden = self.sigmoid(output_vector1)

        output_vector2 = np.dot(
            self.weights2, output_vector_hidden)
        output_vector_network = self.sigmoid(output_vector2)

        self.weights += np.dot(self.Norm_inputs.T, delta)
        # update the weights:
        tmp = self.error * output_vector_network * \
              (1.0 - output_vector_network)

        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)

        self.weights2 += np.dot


        # calculate hidden errors:
        hidden_errors = np.dot(self.weights2.T, self.error)

        # update the weights:
        tmp = hidden * output_vector_hidden * \
              (1.0 - output_vector_hidden)
        self.weights1 += self.learning_rate * \
                         np.dot(tmp, Norm_inputs.T)

        return self.weights1, self.weights2

    def train(self, epochs = 2500):
        for epoch in range(epochs):
            #flow forward and produce an output
            self.feed_forward()
            #go back to the network to make corrections based on the output
            self.backpropagation()
            #keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    def print_matrices(self):
        print("Layer 1: ", self.weights1)
        print("Layer 2: ", self.weights2)

    #predict function
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

#make a neural network
NN = NeuralNet(Norm_inputs, Norm_outputs, learning_rate)

#train the network
#example = np.array([[0.2, 0.3, 0.2]])
#example2 = np.array([[0.1, 0.1, 0.1]])

print("The output: \n")
updated_weights1 ,updated_weights2 = NN.backpropagation()
print(updated_weights1, "\n", updated_weights2)

#print(NN.weights2)


#predict the correct answers
#print(NN.predict(example), ' - Correct: ', example[0][0])
#print(NN.predict(example2), ' - Correct: ', example2[0][0])

#plot the error over the entire training
#plt.figure(figsize=(15,5))
#plt.plot(NN.epoch_list, NN.error_history)
#plt.xlabel('Epoch')
#plt.ylabel('Error')
#plt.show()
