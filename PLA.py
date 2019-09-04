import numpy as np
data = [[0,5],[2,6],[1,7],
             [3,-1],[1,-5],[8,9]]
labels = [0,0,1,1,0,0]

print("Hello World!")

class PLA(object):


    def __init__(self,epochs,alpha,inputs):
    
        self.epochs = 100
        self.alpha = 0.01
        self.weights = np.zeros(inputs + 1)
        print("In Init")


    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation >= 0:
            activation = 1
        else:
            activation = 0
        print("In Predict")
        return activation

    def train(self, inputs, true_outputs):
        for _ in range(self.epochs):
            for row, true_value in zip(inputs, true_outputs):
                prediction = self.predict(row)
                self.weights[1:] = self.weights[1:] + self.alpha * (true_value - prediction)
                self.weights[0] = self.alpha * (true_value - prediction)
                print(self.weights)

    object.train(data,labels)