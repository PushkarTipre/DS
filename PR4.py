import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[2,4],[4,3],[5,6],[7,5],[9,4],[6,2],[8,3]])
y_train = np.array([1,1,1,1,-1,-1,-1])

weights = np.random.rand(2)
bias = np.random.rand()

learning_rate = 0.1

for _ in range(100):
    for inputs, label in zip(x_train,y_train):
        summation = np.dot(weights, inputs)+bias
        activation = 1 if summation >= 0 else -1
        weights += learning_rate * (label - activation) * inputs
        bias += learning_rate * (label - activation)

plt.figure(figsize=(8,6))
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)

x = np.linspace(0,10,100)
y= -(weights[0] * x + bias) / weights[1]
plt.plot(x,y, color='red',label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Region')
plt.legend()
plt.grid(True)
plt.show()

#AND 0 0 0 1
#NOR 1 0 0 0
#OR  0 1 1 1
#NOT x_train = np.array([[0], [1]])
#y_train = np.array([1, 0])
#plt.scatter(x_train[:, 0], np.zeros_like(y_train), c=y_train)
        
        
