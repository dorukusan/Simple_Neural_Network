import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Почему производная такая странная
def sigmoid_derivative(x):
    return x * (1 - x)

input_size = 2
hidden_size = 12
output_size = 1


# Инициализация весов (всего 3 слоя?)
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Инициализация смещений
B1 = np.zeros((1, hidden_size))
B2 = np.zeros((1, output_size))

neural_network = {
    'W1': W1,
    'W2': W2,
    'B1': B1,
    'B2': B2,
}

def forward(X, neural_network):
    # Прямое распространение
    z1 = np.dot(X, W1) + B1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + B2
    y_hat = sigmoid(z2)
    neural_network['z1'] = z1
    neural_network['a1'] = a1
    neural_network['z2'] = z2
    neural_network['y_hat'] = y_hat
    return y_hat


def backward(X, y, y_hat, learning_rate, neural_network):
    # Обратное распространение
    delta2 = (y_hat - y) * sigmoid_derivative(y_hat)
    dW2 = np.dot(neural_network['a1'].T, delta2)
    dB2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = np.dot(delta2, neural_network['W2'].T) * sigmoid_derivative(neural_network['a1'])
    dW1 = np.dot(X.T, delta1)
    dB1 = np.sum(delta1, axis=0)

    # Обновление весов и смещений
    neural_network['W2'] -= learning_rate * dW2
    neural_network['B2'] -= learning_rate * dB2
    neural_network['W1'] -= learning_rate * dW1
    neural_network['B1'] -= learning_rate * dB1


def train(X, y, epochs, learning_rate, neural_network):
    for epoch in range(epochs):
        # Прямое и обратное распространение
        y_hat = forward(X, neural_network)
        backward(X, y, y_hat, learning_rate, neural_network)

        # Вычисление функции потерь (в данном случае квадратичная ошибка)
        loss = np.mean((y_hat - y) ** 2)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.5f}")



def predict(X, neural_network):
    # Прямое распространение для предсказания
    y_hat = forward(X, neural_network)

    # Округление до ближайшего целого значения
    predictions = np.round(y_hat)

    return predictions

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

train(X, Y, 5000, 0.5, neural_network)

predictions = predict(X, neural_network)
print("Predictions:", *predictions)
