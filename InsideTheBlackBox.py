import math
# data is efficacy (1 yes, 0 no), dosage (float)
# note I don't know the actual values he used so I am just guessing
data = [[0,0],[0,.1],[0,.03],
        [1,0.45],[1,.5],[1,.52],
        [0,0.8],[0,.9],[0,1]]

#weights are multiplication (dosage function 0, dosage function 1, effication for 0 and 1)
weights = [-34.4,-2.52,-1.3,2.28]
#biases are added (dosage function 0 dosage function 1, and efficy function)
baises = [2.24,1.29,-.58]


# out activation function is softplus
def activationFunction(x):
    return math.log(1+math.pow(math.e,x))
    
# blue function the values are found by using back propagation
def dosageFunction(x,i):
    return x * weights[i] + baises[i]

def effiactyFunction(x,y):
    return sum([activationFunction(x)*weights[2],activationFunction(y)*weights[3]]) + baises[2]
    
def main():
    dosage = .5
    x = dosageFunction(dosage,0)
    y = dosageFunction(dosage,1)
    efficacy = effiactyFunction(x,y)
    print(efficacy)
    print("Double Bam!!!")

if __name__ == "__main__":
    main()
