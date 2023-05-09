# https://youtu.be/KpKog-L9veg?t=520
# StatQuest ArgMax/SoftMax part 5 Neural Networks
import math

# sets the max value to 1, all others are zero
def argMax(*args):
    return [int(x is max(args)) for x in args]

# can use the derivitive with this because it is smooth
def softMax(*args):
    denominator = sum([math.pow(math.e,x) for x in args])
    return [math.pow(math.e,x)/denominator for x in args]

def main():    
    print(f"softmax: {softMax(1.43, -.4, 0.23)}")
    print(f"argmax: {argMax(1.43, -.4, 0.23)}")

if __name__ == "__main__":
    main()
