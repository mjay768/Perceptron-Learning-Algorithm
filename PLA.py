import numpy as np


print("Hello World!")
class PLA(object):
    data = [[0, 5], [2, 6], [1, 7],
            [3, -1], [1, -5], [8, 9]]
    labels = [0, 0, 1, 1, 0, 0]


    def __init__(self,epochs,alpha,inputs):

        #self.epochs = 100
        #self.alpha = 0.01
        self.weights = np.zeros(inputs + 1)
        print("Hi there!")
        self.train(self.data,self.labels)


    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        #print(summation)
        if summation >= 0:
            activation = 1
        else:
            activation = 0

        return activation

    def train(self, inputs, true_outputs):
        for _ in range(self.epochs):
            for row, true_value in zip(inputs, true_outputs):
                prediction = self.predict(row)
                self.weights[1:] = self.alpha * (true_value - prediction) + self.weights[1:]
                self.weights[0] = self.alpha * (true_value - prediction)
                #print(self.weights)

pla = PLA(epochs=100,alpha=0.01,inputs=2)
