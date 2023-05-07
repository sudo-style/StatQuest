import math
import sympy as sp

def main():
    line = Line()
    line.gradientDescent()


class Line:
    def __init__(self, slope = 0.64, intercept = 0.0):
        self.slope = slope
        self.intercept = intercept
        
        # observed (weight, height) for each data point
        self.data = [[0.5,1.4],[2.3,1.9],[2.9,3.2]]

        self.previousStepSize = 1.0
        self.MIN_STEP_SIZE = 0.01
        self.LEARNING_RATE = 0.1
        self.MAX_RECURSIONS = 10

    # predicts the height given the the known weight
    def predictedHeight(self, weight):
        return self.intercept + (self.slope * weight)    

    # loss function
    def SSResiduals(self, total = 0):
        for weight, height in self.data:
            predictedHeight = self.predictedHeight(weight)
            total += math.pow(height - predictedHeight,2)
        return total
    
    # new shit can't belive how easy it is to implement this    
    def derivitiveOfSSR(self, total = 0):
        intercept,slope = sp.symbols('intercept slope')
        for weight,height in self.data:
            total += (height - (intercept + (self.slope * weight)))**2
        fbyIntercept_prime = sp.diff(total, intercept)
        return fbyIntercept_prime.subs(intercept,self.intercept) # bet that this is the issue

    def print(self):
        print(f"slope:{self.slope}")
        print(f"step size:{self.previousStepSize}")
        print(f"intercept:{self.intercept}\n")
        print()
        
    def gradientDescent(self, totalRecursions = 0):
        if (totalRecursions > self.MAX_RECURSIONS):
            print("MAX_RECURSIONS")
            return
        if (abs(self.previousStepSize) < self.MIN_STEP_SIZE):
            print("MINIMUM STEP SIZE")
            return

        # stores the slope of the ssr
        slopeOfSSR = self.derivitiveOfSSR()
        # gets how big the step is
        stepsize = slopeOfSSR * self.LEARNING_RATE
        # it will continue one after the min value is hit
        self.previousStepSize = stepsize
        # update the new intercept 
        self.intercept = self.intercept - stepsize
        self.print()
        # recurse through adding one to the max calls
        self.gradientDescent(totalRecursions + 1)

        
if __name__ == "__main__":
    main()
