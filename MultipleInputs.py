# https://youtu.be/83LYR-1IcjA?t=764
# Neural Netwroks Pt. 4 - StatQuest

def main():
    petalWidth = .5
    septalWidth = .37
    
    n1 = Neuron(1.6,-2.5,0.6)
    n2 = Neuron(0.7,-1.5,0.4)

    blue = n1.input(petalWidth,septalWidth)
    orange = n2.input(petalWidth,septalWidth)

    blueReLu = reLu(blue)
    orangeReLu = reLu(orange)

    n3 = Neuron(0, -0.1,1.5)
    n4 = Neuron(0, 2.4, -5.2)
    n5 = Neuron(1, -2.2, 3.7)

    setosa = n3.input(blueReLu, orangeReLu)
    versicolor = n4.input(blueReLu, orangeReLu)
    virginica = n5.input(blueReLu, orangeReLu)

    print(f"setosa: {setosa}")
    print(f"versicolor: {versicolor}")
    print(f"virginica: {virginica}")

class Neuron:
    def __init__(self, bias, *weights):
        self.bias = bias
        self.weights = [*weights]
        
    def print(self):
        print(f"bias: {self.bias}")
        print(f"weights: {self.weights}")

    def input(self, petalWidth, septalWidth):
        return petalWidth * self.weights[0] + septalWidth * self.weights[1] +self.bias

def reLu(x):
    return max(0,x)

if __name__ == "__main__":
    main()
