import numpy as np

class Network(object):

    def __init__(self, sizes, momentum=0.9):
        # Initialization remains the same
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # Momentum-related initialization
        self.momentum = momentum
        self.velocity_b = [np.zeros(b.shape) for b in self.biases]
        self.velocity_w = [np.zeros(w.shape) for w in self.weights]

    def update_weights(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent with momentum to a single mini batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Compute gradients for the mini-batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.epoch(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update velocities and apply updates with momentum
        self.velocity_w = [self.momentum * vw - (eta / len(mini_batch)) * nw for vw, nw in zip(self.velocity_w, nabla_w)]
        self.velocity_b = [self.momentum * vb - (eta / len(mini_batch)) * nb for vb, nb in zip(self.velocity_b, nabla_b)]

        # Update weights and biases
        self.weights = [w + vw for w, vw in zip(self.weights, self.velocity_w)]
        self.biases = [b + vb for b, vb in zip(self.biases, self.velocity_b)]


    def epoch(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    

    can you create a function that goes through a 2d numpy dataset where the first dimension is examples and the second dimension is features. Then 

