import math
# todo make this a class

# observed weight, height
data = [[0.5,1.4],[2.3,1.9],[2.9,3.2]]
    
def main():
    gradientDescent()

# predicted height = intercept + slope * weight
def predictedHeightFunction(weight,intercept = 0, slope = 0.64):
    return intercept + (slope * weight)

# resididual = observedHeight - predictedheight
def residual(n,intercept = 0, slope = 0.64):
    observedWeight = data[n][0]
    predictedHeight = predictedHeightFunction(observedWeight,intercept)
    observedHeight = data[n][1]
    return observedHeight - predictedHeight    

def sumSqurareResiduals(intercept = 0, slope = 0.64):
    residuals = [math.pow(residual(n,intercept),2) for n in range (len(data))]
    print(f"residuals: {residuals}")
    print(f"intercept: {intercept}")
    return round(sum(residuals),2)

# this function was found, by taking the deravitive of SSR
# aka derivitive of the loss function
def derivitiveOfSSR(intercept = 0, slope = 0.64, total = 0):
    for n in range(len(data)):
        weight = data[n][0]
        height = data[n][1]

        # need to learn more to figure out how to generalize this function
        total += -2*(height - (intercept + slope * weight)) 
    return round(total,2)

# if too small it will take a long time to learn, if too big may skip over the valley
def stepSize(slope, learningRate = 0.1):
    return slope * learningRate

# calculates the new intercept given the old intercept and stepsize
def newIntercept(oldIntercept, stepSize):
    return oldIntercept - stepSize

# recursive function that will keep going until it finds the smallest SSR
# thus the new Intercept will be the value we are looking for 
def gradientDescent(oldIntercept = 0, previousStep = 1.0, minStepSize = 0.01, maxRecursion = 1000):
    if (maxRecursion < 0):
        print("MAX RECURSION")
        return
    if (abs(previousStep) < minStepSize):
        print("MINIMUM STEP SIZE")
        return
    
    slope = derivitiveOfSSR(oldIntercept)
    stepSize_ = stepSize(slope)
    newIntercept_ = newIntercept(oldIntercept, stepSize_)
    print(f"slope: {slope}")
    print(f"step size: {stepSize_}")
    print(f"new intercept: {newIntercept_}")
    gradientDescent(newIntercept_, stepSize_, minStepSize, maxRecursion - 1)
    
if __name__ == "__main__":
    main()
