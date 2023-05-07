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
        self.MIN_STEP_SIZE = 0.00001
        self.LEARNING_RATE = 0.01
        self.MAX_RECURSIONS = 950

        # some other functions I need apparently
        self.stepsizeSlope = 1.0
        self.stepsizeIntercept = 1.0
        self.total = 0


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


    # these functions are too similar
    def interceptStep(self):
        # take the gradient of the loss function
        dirivitiveIntercept = sp.diff(self.total, self.dintercept)     
        # this slope tells in what direction and by how much I can step
        # plug the parameter values into the derivatives/gradient
        slopeOfIntercept = dirivitiveIntercept.subs(self.subsituitutions)
        # calculate the step sizes
        self.stepsizeIntercept = slopeOfIntercept * self.LEARNING_RATE
        self.intercept = self.intercept - self.stepsizeIntercept

    def slopeStep(self):
        # take the gradient of the loss function        
        dirivitiveSlope = sp.diff(self.total, self.dslope)
        # this slope tells in what direction and by how much I can step
        # plug the parameter values into the derivatives/gradient
        slopeOfSlope = dirivitiveSlope.subs(self.subsituitutions)
        # calculate the step sizes
        self.stepsizeSlope = slopeOfSlope * self.LEARNING_RATE        
        self.slope = self.slope - self.stepsizeSlope

    # this adds up all of the functions in terms of variables
    def sumVars(self):
        self.total = 0
        self.dintercept,self.dslope = sp.symbols('intercept slope')
        for weight,height in self.data:
            self.total += (height - (self.dintercept + (self.dslope * weight)))**2
        self.subsituitutions = {self.dintercept: self.intercept, self.dslope: self.slope}

    # my question is, should each of these be their own function for one step? I think so, for one step yes
    def gradientDescent(self, totalRecursions = 0):
        if (totalRecursions > self.MAX_RECURSIONS):
            print("MAX_RECURSIONS")
            return

        if (abs(self.stepsizeSlope) < self.MIN_STEP_SIZE and abs (self.stepsizeIntercept) < self.MIN_STEP_SIZE):
            print("MINIMUM STEP SIZE")
            return 

        self.sumVars() 
        self.interceptStep()
        self.slopeStep()
        
        # recurse through adding one to the max calls
        self.gradientDescent(totalRecursions + 1)
        #print(totalRecursions)
        
if __name__ == "__main__":
    main()
