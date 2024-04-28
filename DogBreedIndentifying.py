import numpy as np
import tarfile
from PIL import Image
import os

class CNN:
    def __init__(self, num_classes):
        # Initialize parameters with random values for convolutional and fully connected layers
        self.params = {
            'W1': np.random.randn(3, 3, 3, 8) * 0.1,  # Convolutional layer 1 weights
            'W2': np.random.randn(3, 3, 8, 16) * 0.1, # Convolutional layer 2 weights
            'W3': np.random.randn(1024, 128) * 0.1,  # Fully connected layer weights
            'W4': np.random.randn(128, num_classes) * 0.1  # Output layer weights
        }
        self.biases = {
            'b1': np.zeros(8),
            'b2': np.zeros(16),
            'b3': np.zeros(128),
            'b4': np.zeros(num_classes)
        }

    def conv_forward(self, A_prev, W, b, stride=1, pad=1):
        # Perform a forward pass through a convolutional layer
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f, _, _, n_C = W.shape
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
        Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        
        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                        Z[i, h, w, c] = np.sum(a_slice * W[..., c]) + b[c]
        return Z, (A_prev, W, b, stride, pad)  # Return both the result and cache for backpropagation

    def relu(self, Z):
        # Apply ReLU activation function
        return np.maximum(0, Z), Z  # Return the activated values and cache Z

    def pool_forward(self, A_prev, f=2, stride=2):
        # Perform a forward pass through a pooling layer
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        n_H = int((n_H_prev - f) / stride + 1)
        n_W = int((n_W_prev - f) / stride + 1)
        n_C = n_C_prev
        A = np.zeros((m, n_H, n_W, n_C))
        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        A[i, h, w, c] = np.max(A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c])
        return A, (A_prev, f, stride)  # Also return cache for backpropagation

    def fc_forward(self, A_prev, W, b):
        # Perform a forward pass through a fully connected layer
        Z = A_prev.reshape(A_prev.shape[0], -1).dot(W) + b
        return Z, (A_prev, W, b)  # Return the result and cache

    def softmax_loss(self, A, Y):
        # Compute softmax loss for classification
        m = Y.shape[0]
        exps = np.exp(A - np.max(A, axis=1, keepdims=True))
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        log_likelihood = -np.log(softmax[np.arange(m), Y])
        loss = np.sum(log_likelihood) / m
        dA = softmax
        dA[np.arange(m), Y] -= 1
        dA /= m
        return loss, dA

    def predict(self, X):
        # Predict the class of the input data
        Z1, _ = self.conv_forward(X, self.params['W1'], self.biases['b1'])
        A1, _ = self.relu(Z1)
        P1, _ = self.pool_forward(A1)
        Z2, _ = self.conv_forward(P1, self.params['W2'], self.biases['b2'])
        A2, _ = self.relu(Z2)
        P2, _ = self.pool_forward(A2)
        Z3, _ = self.fc_forward(P2, self.params['W3'], self.biases['b3'])
        A3, _ = self.relu(Z3)
        Z4, _ = self.fc_forward(A3, self.params['W4'], self.biases['b4'])
        predictions = np.argmax(Z4, axis=1)
        return predictions

    def train(self, X, Y, epochs, lr, batch_size=32):
        # Train the model using mini-batch gradient descent
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        for epoch in range(epochs):
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, X.shape[0])
                batch_X, batch_Y = X[batch_start:batch_end], Y[batch_start:batch_end]

                # Forward pass
                Z1, cache1 = self.conv_forward(batch_X, self.params['W1'], self.biases['b1'])
                A1, cache1_relu = self.relu(Z1)
                P1, cache1_pool = self.pool_forward(A1)
                Z2, cache2 = self.conv_forward(P1, self.params['W2'], self.biases['b2'])
                A2, cache2_relu = self.relu(Z2)
                P2, cache2_pool = self.pool_forward(A2)
                Z3, cache3 = self.fc_forward(P2, self.params['W3'], self.biases['b3'])
                A3, cache3_relu = self.relu(Z3)
                Z4, cache4 = self.fc_forward(A3, self.params['W4'], self.biases['b4'])
                loss, dZ4 = self.softmax_loss(Z4, batch_Y)

                # Backpropagation
                dA3, dW4, db4 = self.fc_backward(dZ4, cache4)
                dZ3 = self.relu_backward(dA3, cache3_relu)
                dP2 = self.pool_backward(dZ3, cache2_pool)
                dA2 = self.relu_backward(dP2, cache2_relu)
                dZ2 = self.conv_backward(dA2, cache2)
                dP1 = self.pool_backward(dZ2, cache1_pool)
                dA1 = self.relu_backward(dP1, cache1_relu)
                _, dW1, db1 = self.conv_backward(dA1, cache1)

                # Gradient update
                self.update_params({'dW1': dW1, 'dW2': dW4, 'dW3': dW3, 'dW4': dW4,
                                    'db1': db1, 'db2': db2, 'db3': db3, 'db4': db4}, lr)

                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, Loss: {loss}")

    def update_params(self, grads, lr):
        # Update the parameters of the network
        for key, value in self.params.items():
            self.params[key] -= lr * grads['d' + key]
        for key, value in self.biases.items():
            self.biases[key] -= lr * grads['db' + key]

def load_images_and_labels(images_path, annotations_path):
    # Load images and labels for training
    images = []
    labels = []
    with open(annotations_path, 'r') as file:
        for line in file:
            image_name, label = line.strip().split()
            image_path = os.path.join(images_path, image_name + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image = image.resize((32, 32))
            images.append(np.array(image) / 255.0)  # Normalize images
            labels.append(int(label))
    return np.array(images), np.array(labels)

def main():
    # Main function to execute training and prediction
    images_path = 'images'
    annotations_path = 'annotations/trainval.txt'
    X_train, Y_train = load_images_and_labels(images_path, annotations_path)
    num_classes = len(set(Y_train))  # Number of classes
    model = CNN(num_classes)
    model.train(X_train, Y_train, epochs=10, lr=0.01)

    #Predicting a new image
    image_path = 'path/to/testimage.jpg'
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)
    print(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()
