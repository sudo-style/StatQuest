import numpy as np
import math
from ActivationFunctions import softPlus
from ActivationFunctions import relu

def main():
    nnDosage = NeuralNetwork(1,1,'sp',2)
    nnDosage.biases = [2.14, 1.29, -0.58] # using determined biases and weights to make sure they match the target
    nnDosage.weights = [-34.4, -2.52, -1.3, 2.28]
    print(nnDosage.output(0.5))
    print("target: [1.03]\n")
    
    nnFlowers = NeuralNetwork(2,3,'relu',2)
    nnFlowers.weights = [-2.5, -1.5, 0.6, 0.4, -.1, 2.4, -2.2, 1.5, -5.2, 3.7]
    nnFlowers.biases = [1.6,0.7,0,0,1]
    print(nnFlowers.output(0.5,.37))
    print("target: [0.09, 0.86, 0.10]\n")

    nnRandom = NeuralNetwork(1,1,'sp',4,3,4)    
    print(f"weights: {nnRandom.weights}")
    print(f"numWeights: {len(nnRandom.weights)}")

    
    print(f"biases: {nnRandom.biases}")
    print(f"numWeights: {len(nnRandom.biases)}")
    print(f"randomOutput: {nnRandom.output(.43)}")

class NeuralNetwork(): # makes a neural network with random values populated, needs to be trained
    def __init__(self, inputs, outputs, funcType, *hiddenLayers):
        self.inputs = inputs
        self.outputs = outputs
        self.funcType = funcType
        self.hiddenLayers = hiddenLayers
        self.fullNetwork = [self.inputs] + list(self.hiddenLayers) + [self.outputs]
        self.biases = [0.0] * self.calcNumBiases() # bias values start at zero
        self.weights = np.random.rand(self.calcNumWeights()) # give random values for weights

        self.weights = [0.34523800, 0.63133228, 0.75229619, 0.78843551, 0.31813906, 0.42814637,
                        0.84948657, 0.12426702, 0.89754243, 0.74264793, 0.18185164, 0.67522291,
                        0.77942494, 0.26823629, 0.96129143, 0.05732444, 0.28585249, 0.30001379,
                        0.73170743, 0.46273547, 0.88589686, 0.71097702, 0.51699631, 0.60683042,
                        0.33108003, 0.92519579, 0.83285673, 0.28336363, 0.61487147, 0.55821392,
                        0.707886,   0.71905211]
        self.biases = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # should be 8.65 ish

    def calcNumBiases(self): # hidden layers + outputs = numBiases 
        return sum(self.fullNetwork[1:])
    
    '''
        # multiplies adjcent numbers and sums them
        2 input, [2,3] hidden, 2 output
        = 2*2 + 2*3 + 3*2 = 16
        return 16
    '''
    def calcNumWeights(self): 
        fullNetwork = self.fullNetwork 
        return sum([x * y for x, y in zip(fullNetwork, fullNetwork[1:])])

    # picks the activation function that is requested
    def activationFunction(self, x):
        if self.funcType == 'relu': return relu(x)
        if self.funcType == 'sp': return softPlus(x)
        return None

    def inputToHidden(self,inputs):
        layers = 1
        shiftWeight = self.shiftWeight(layers)   
        shiftBias = self.shiftBias(layers)       
        fullNetwork = self.fullNetwork      
        if len(inputs) != fullNetwork[0]: return None
        sums = self.biases
        for i in range(self.layers(layers-1)): 
            sums[shiftBias + i%self.fullNetwork[layers]] += inputs[int(i/fullNetwork[layers])]*self.weights[shiftWeight+i]
        return sums

    # the reason these have to be different is because the input changes the sums
    def hiddenToHidden(self, sums, layers): 
        shiftWeight = self.shiftWeight(layers) # I don't want any magical constants, except for testing
        shiftBias = self.shiftBias(layers)
        fullNetwork = self.fullNetwork 
        for i in range(self.layers(layers-1)): # layers -1 means the previous layer
            sums[shiftBias+i%fullNetwork[layers]] += self.activationFunction(sums[int(i/fullNetwork[layers])]) * self.weights[shiftWeight+i]
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
        print(inputs)
        outputs = self.outputs
        fullNetwork = self.fullNetwork
        if len(inputs) != fullNetwork[0]: return None

        # do a summation on the weights, biases of the first input layer to hidden layer
        sums = self.inputToHidden(inputs)
        # hidden to hidden takes care of all of the layers that need an activation layer including the outputs
        print(sums)
        for i in range(len(self.hiddenLayers)): 
            sums = self.hiddenToHidden(sums,i+2) # 1 is the first layer which was already added i is zero on first pass
            print(sums)
        return sums[-1 * outputs:] #the end of sums are the outputs, if there are outputs then sum[x,x,x,x,a,b,c] a,b,c is the outputs

if __name__ == "__main__":
    main()
