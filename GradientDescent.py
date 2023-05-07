import math
import sympy as sp

def main():
    line = Line()
    line.gradientDescent()
    line.print()


class Line:
    def __init__(self, slope = 1.0, intercept = 0.0):
        self.slope = slope
        self.intercept = intercept
        
        # observed (weight, height) for each data point
        self.data = [[0.5,1.4],[2.3,1.9],[2.9,3.2]]


        self.previousStepSizeIntercept = 0.0
        self.previousStepSizeSlope = 1.0
        self.MIN_STEP_SIZE = 0.0001
        self.LEARNING_RATE = 0.01
        self.MAX_RECURSIONS = 1000

    # predicts the height given the the known weight
    def predictedHeight(self, weight):
        return self.intercept + (self.slope * weight)    

    # loss function
    def SSResiduals(self, total = 0):
        for weight, height in self.data:
            predictedHeight = self.predictedHeight(weight)
            total += math.pow(height - predictedHeight,2)
        return total
    
    def print(self):
        print(f"intercept:{self.intercept}")
        print(f"slope:{self.slope}")
        print(f"step size Intercept:{self.previousStepSizeIntercept}")
        print(f"step size Slope:{self.previousStepSizeSlope}\n")
        print()

    # my question is, should each of these be their own function for one step? I think so, for one step yes
    def gradientDescent(self, totalRecursions = 0):
        if (totalRecursions > self.MAX_RECURSIONS):
            print("MAX_RECURSIONS")
            return

        total = 0
        intercept,slope = sp.symbols('intercept slope')
        for weight,height in self.data:
            total += (height - (intercept + (slope * weight)))**2
                
        # take the gradient of the loss function
        dirivitiveIntercept = sp.diff(total, intercept)#= self.derivitiveOfSSRIntercept()
        dirivitiveSlope = sp.diff(total, slope)
        
        # this slope tells in what direction and by how much I can step
        # plug the parameter values into the derivatives/gradient
        subsituitutions = {intercept: self.intercept, slope: self.slope}
        slopeOfIntercept = dirivitiveIntercept.subs(subsituitutions)
        slopeOfSlope = dirivitiveSlope.subs(subsituitutions)

        # calculate the step sizes
        stepsizeIntercept = slopeOfIntercept * self.LEARNING_RATE
        stepsizeSlope = slopeOfSlope * self.LEARNING_RATE
        

        self.previousStepSizeIntercept = stepsizeIntercept
        self.previousStepSizeSlope = stepsizeSlope
        
        # calculate the new parameters
        self.intercept = self.intercept - stepsizeIntercept
        self.slope = self.slope - stepsizeSlope
        
        if (abs(stepsizeSlope) < self.MIN_STEP_SIZE and abs (stepsizeIntercept) < self.MIN_STEP_SIZE):
            print("MINIMUM STEP SIZE")
            return 
        
        # recurse through adding one to the max calls
        self.gradientDescent(totalRecursions + 1)

        
if __name__ == "__main__":
    main()
