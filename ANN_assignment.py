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

class NeuralNet:
    #variables
    def __init__(self, Norm_inputs, Norm_outputs):
        self.Norm_inputs = Norm_inputs
        self.Norm_outputs = Norm_outputs

        #weights
        self.weights = np.array([ [.50], [.50], [.50] ])
        self.error_history = []
        self.epoch_list = []

    # calculate the sigmoid
    def sigmoid(self, x, deriv = False):
        if deriv == True:
            return x * (1-x)
        return 1 / (1 + np.exp(-x))

    #data will flow through the neural network
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.Norm_inputs, self.weights))

    def backpropagation(self):
        self.error = self.Norm_outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.Norm_inputs.T, delta)

    def train(self, epochs = 14):
        for epoch in range(epochs):
            #flow forward and produce an output
            self.feed_forward()
            #go back to the network to make corrections based on the output
            self.backpropagation()
            #keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    #predict function
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

#make a neural network
NN = NeuralNet(Norm_inputs, Norm_outputs)

#train the network
example = np.array([[0.2, 0.3, 0.2]])
example2 = np.array([[0.1, 0.1, 0.1]])

#predict the correct answers
print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example2), ' - Correct: ', example2[0][0])

#plot the error over the entire training
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()