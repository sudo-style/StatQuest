import numpy as np
import math

def main():
    nn = NeuralNetwork(2,3,'relu',2) # this will only do random values
    print(nn.output(0.3,2.3))
    

class NeuralNetwork(): # makes a neural network with random values populated, needs to be trained
    def __init__(self, inputs, outputs, funcType, *hiddenLayers):
        self.inputs = inputs
        self.outputs = outputs
        self.funcType = funcType
        self.hiddenLayers = hiddenLayers
        self.fullNetwork = [self.inputs] + list(self.hiddenLayers) + [self.outputs]
        
        self.biases = startBiases(self.calcNumBiases())
        self.weights = randomWeights(self.calcNumWeights())

        '''main():
        nn = nn = NeuralNetwork(2,3,'relu',2)
        print(nn.output(0.5,.37))
        self.weights = [-2.5, -1.5, 0.6, 0.4, -.1, 2.4, -2.2, 1.5, -5.2, 3.7]
        self.biases = [1.6,0.7,0,0,1]
        # should output this [0.08979999999999991, 0.8632000000000003, 0.10419999999999963]'''
        
        '''main():
        nn = NeuralNetwork(1,2,'sp',2)
        print(nn.output(0.5))
        self.biases = [2.14, 1.29, -0.58]
        self.weights = [-34.4, -2.52, -1.3, 2.28]
        should output this [1.03]'''

    def calcNumBiases(self):
        return sum(self.hiddenLayers) + self.outputs

    def calcNumWeights(self):
        fullNetwork = self.fullNetwork
        total = 0
        for i in range(len(fullNetwork)-1):
            total += fullNetwork[i] * fullNetwork[i+1]
        return total

    def activationFunction(self, x):
        if self.funcType == 'relu': return relu(x)
        if self.funcType == 'sp': return softPlus(x)
        return None
    
    def hiddenToHidden(self, sums, layers): 
        shiftWeight = self.shiftWeight(layers) # I don't want any magical constants, except for testing
        shiftBias = self.shiftBias(layers)
        fullNetwork = self.fullNetwork 
        n = 0
        for i in range(self.layers(layers-1)): # layers -1 is the previous layer
            sums[shiftBias+i%fullNetwork[layers]] += self.activationFunction(sums[int(n/fullNetwork[layers])]) * self.weights[shiftWeight+i]
            n+=1
        return sums

    def inputToHidden(self,inputs):
        fullNetwork = self.fullNetwork
        if len(inputs) != fullNetwork[0]: return None
        sums = self.biases
        n = 0
        for i in range(self.layers(0)):
            sums[i%2] += inputs[int(n/fullNetwork[1])]*self.weights[i]
            n += 1
        return sums

    def layers(self,x):
        fullNetwork = self.fullNetwork
        if x < 0: return None
        if (x > len(fullNetwork)-2): return None
        return fullNetwork[x] * fullNetwork[x+1]

    # depending on how many layers deep we are will determine the shift bias
    # we can uses these so we can figure out which bias we will add
    def shiftBias(self,layers):
        return sum(self.fullNetwork[1:layers])

    # depending on how many layers deep we are will determine the shift weight
    # we can uses these so we can figure out which weight we will multiply
    def shiftWeight(self,layers):
        fullNetwork = [0] + self.fullNetwork
        multipliedNetwork = [fullNetwork[i] * fullNetwork[i+1] for i in range(len(fullNetwork)-1)]
        return sum(multipliedNetwork[0:layers])

    def output(self, *args):
        inputs = args        
        outputs = self.outputs
        fullNetwork = self.fullNetwork
        if len(inputs) != fullNetwork[0]: return None

        # do a summation on the weights, biases of the first input layer to hidden layer
        sums = self.inputToHidden(inputs)
        # hidden to hidden takes care of all of the layers that need an activation layer including the outputs
        sums = self.hiddenToHidden(sums,2)
        return sums[-1 * outputs:]
    
def startBiases(n):
    return [0] * n

def randomWeights(n):
    return np.random.rand(n)

def relu(x):
    return max(0,x)

def softPlus(x):
    return math.log(1+math.pow(math.e,x))    

if __name__ == "__main__":
    main()
