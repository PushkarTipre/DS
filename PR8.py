import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        self.weights = sum(np.outer(pattern, pattern) for pattern in patterns) / len(patterns)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, max_iter=100):
        for _ in range(max_iter):
            prev_pattern = pattern.copy()
            pattern = np.sign(np.dot(pattern, self.weights))
            if np.array_equal(pattern, prev_pattern):
                break
        return pattern

# Take user input for defining patterns
patterns = []
while len(patterns) < 4:
    user_input = input(f"Enter pattern {len(patterns) + 1} as a sequence of 1s and -1s: ")
    if all(char in {'1', '-', ' '} for char in user_input):
        patterns.append([int(char) for char in user_input.split()])
    else:
        print("Invalid input. Please enter a sequence of 1s and -1s.")

# Create and train the Hopfield Network
hopfield_net = HopfieldNetwork(len(patterns[0]))
hopfield_net.train(patterns)

# Test the network with some noisy inputs
test_inputs = [[1, -1, 1, -1], [1, -1, 1, 1], [1, 1, 1, 1], [-1, -1, -1, -1]]
for input_pattern in test_inputs:
    print("Input Pattern:", input_pattern)
    print("Predicted Pattern:", hopfield_net.predict(input_pattern))
    print()
