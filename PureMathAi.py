class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        # He initialization for ReLU layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_onehot):
        m = X.shape[0]
        # Output layer gradients
        dz2 = self.a2 - y_onehot
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_grad(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        # Update parameters
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs, batch_size, verbose=True):
        num_samples = X.shape[0]
        num_classes = self.W2.shape[1]
        # One‑hot encode labels
        y_onehot = np.eye(num_classes)[y]
        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(num_samples)
            X_shuff = X[perm]
            y_shuff = y_onehot[perm]
            # Mini‑batch training
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuff[i:i+batch_size]
                y_batch = y_shuff[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            if verbose:
                # Compute training accuracy
                pred = np.argmax(self.forward(X), axis=1)
                acc = np.mean(pred == y)
                print(f'Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}')

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
