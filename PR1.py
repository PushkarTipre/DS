import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def tanh(x):
      return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    exp_values=np.exp(x-np.max(x,axis=0,keepdims=True))
    return exp_values / np.sum(exp_values,axis=0,keepdims=True)

start_range = float(input("Enter the start of the input range: ")) #Input is 5
end_range = float(input("Enter the end of the input range: "))#Input is -5
step = int(input("Enter the number of steps: "))#Input is 100

x = np.linspace(start_range, end_range, step)

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.title("Sigmoid Activation Function")
plt.legend()

plt.subplot(2,2,2)
plt.plot(x, relu(x), label='ReLU')
plt.title("ReLU Activation Function")
plt.legend()

plt.subplot(2,2,3)
plt.plot(x, tanh(x), label='Tanh')
plt.title("Tanh Activation Function")
plt.legend()

plt.subplot(2,2,4)
plt.plot(x, softmax(x), label='Softmax')
plt.title("Softmax Activation Function")
plt.legend()

plt.tight_layout()
plt.show()
