import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная функции активации
def sigmoid_derivative(x):
    return np.exp(-x)/((np.exp(-x)+1)**2)
# Количество нейронов во входном слое
input_size = 784
# Количество нейронов в скрытом слое
hidden_size = 20
# Количество нейронов в выходном слое
output_size = 10


# Инициализация весов
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


def backward(X, y, y_hat, learning_rate, neural_network, z1, z2, a1):
    # Обратное распространение
    # delta2 = (y_hat - y) * sigmoid_derivative(y_hat)
    # dW2 = np.dot(neural_network['a1'].T, delta2)
    # dB2 = np.sum(delta2, axis=0, keepdims=True)
    #
    # delta1 = np.dot(delta2, neural_network['W2'].T) * sigmoid_derivative(neural_network['a1'])
    # dW1 = np.dot(X.T, delta1)
    # dB1 = np.sum(delta1, axis=0)

    dB2 = np.sum(2*(y_hat - y)*sigmoid_derivative(z2), axis=0, keepdims=True)

    dW2 = []
    for i in range(len(neural_network['W2'])):
        temp = 2*(y_hat - y)*sigmoid_derivative(z2)
        for j in range(len(temp)):
            temp[j] = temp[j] * a1[j][i]
        c = np.sum(temp, axis=0, keepdims=True)[0]
        dW2.append(c)
    dW2 = np.array(dW2)

    dB1 = []
    for i in range(len(neural_network['W2'])):
        du = 2 * (y_hat - y) * sigmoid_derivative(z2)
        w = neural_network['W2'][i].T
        for j in range(len(du)):
            h = sigmoid_derivative(z1[j][i])
            du[j] = du[j] * w * h
        du = np.sum(du, axis=1, keepdims=True)
        dB1.append([np.sum(du)])
    dB1 = np.array(dB1).T

    dW1 = np.zeros((len(neural_network['W1']), len(neural_network['W1'][0])))
    for i in range(len(neural_network['W1'])):
        for j in range(len(neural_network['W1'][i])):
            du = 2 * (y_hat - y) * sigmoid_derivative(z2)
            w = neural_network['W2'][j].T
            for k in range(len(du)):
                h = sigmoid_derivative(z1[k][j])
                l = X[k][i]
                du[k] = du[k] * w * h * l
            du = np.sum(du, axis=1, keepdims=True)
            du = np.sum(du)
            dW1[i, j] = du









    # Обновление весов и смещений
    neural_network['W2'] -= learning_rate * dW2
    neural_network['B2'] -= learning_rate * dB2
    neural_network['W1'] -= learning_rate * dW1
    neural_network['B1'] -= learning_rate * dB1


def train(X, y, epochs, learning_rate, neural_network, batch_size):
    for epoch in range(epochs):
        # Прямое и обратное распространение
        y_hat = forward(X, neural_network)
        z1 = neural_network['z1']
        z2 = neural_network['z2']
        a1 = neural_network['a1']
        rng = np.random.default_rng()
        indices = rng.choice(y_hat.shape[0], size=batch_size, replace=False)
        new_X = X[indices]
        new_y_hat = y_hat[indices]
        new_y = y[indices]
        new_z1 = z1[indices]
        new_z2 = z2[indices]
        new_a1 = a1[indices]

        backward(new_X, new_y, new_y_hat, learning_rate, neural_network, new_z1, new_z2, new_a1)

        # Вычисление функции потерь
        loss = (y_hat - y)**2
        loss = np.sum(loss, axis=1)
        loss = np.mean(loss)


        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")



def predict(X, neural_network):
    # Прямое распространение для предсказания
    y_hat = forward(X, neural_network)
    max_index = np.argmax(y_hat)

    return max_index


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:20]
y_test = y_test[:20]
x_train = x_train.astype('float32') / 255
new_x_train = []
for i in range(len(x_train)):
    vector = []
    for j in range(len(x_train[i])):
        for k in range(len(x_train[i][j])):
            vector.append(x_train[i][j][k])
    new_x_train.append(vector)

new_x_test = []
for i in range(len(x_test)):
    vector = []
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            vector.append(x_test[i][j][k])
    new_x_test.append(vector)


new_y_train = []
for i in range(len(y_train)):
    vector = [0.0] * 10
    vector[y_train[i]] = 1
    new_y_train.append(vector)

new_y_test = []
for i in range(len(y_test)):
    vector = [0.0] * 10
    vector[y_test[i]] = 1
    new_y_test.append(vector)

new_x_train = np.array(new_x_train)
new_y_train = np.array(new_y_train)

train(new_x_train, new_y_train, 100, 0.5, neural_network, 10)

for i in range(len(new_x_test)):
    predictions = predict(new_x_test[i], neural_network)
    print("Prediction:", predictions, " Actual:", np.argmax(new_y_test[i]))
