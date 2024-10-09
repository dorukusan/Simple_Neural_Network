import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn import metrics


# Разделение на тренировочные и тестовые данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Срез первых 100 значений
# x_train = x_train[:100]
# y_train = y_train[:100]

# Нормализация данных
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Создание модели
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Преобразуем 2D изображения в 1D векторы
model.add(Dense(784, activation='sigmoid'))  # Скрытый слой с 784 нейронами
model.add(Dense(10, activation='softmax'))   # Выходной слой с 10 нейронами (для классов)

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=10, batch_size=10)

# Прогнозирование
predictions = model.predict(x_test)

# Предсказанные и реальные классы
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Вывод примеров
for i in range(20):
    print(f"Example {i + 1}: Predicted class: {predicted_classes[i]}, Actual class: {actual_classes[i]}")

# Оценка точности
accuracy = metrics.accuracy_score(actual_classes, predicted_classes)
print("Accuracy: ", accuracy)


# Предсказывание введенного в поле числа
def predict_digit(img):
    img = img.resize((28, 28))  # Изменяем размер на 28x28
    img = img.convert('L')       # Конвертируем в градации серого
    img = np.array(img)          # Преобразуем в массив NumPy
    img = img.astype('float32') / 255  # Нормализация
    img = 1 - img

    img[0:4, :] = 0.0
    img[:, 0] = 0.0
    img[:, -1] = 0.0

    img = img.reshape(1, 28, 28)  # Изменяем форму на (1, 28, 28)

    res = model.predict(img)
    digit = np.argmax(res)
    acc = np.max(res)  # Получаем максимальную вероятность
    return digit, acc


# Графическая форма
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="")

    def classify_handwriting(self):
        # Захватываем изображение с холста
        x = self.winfo_x() + self.canvas.winfo_x()
        y = self.winfo_y() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        im = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Проверка на пустое изображение перед предсказанием
        if np.sum(np.array(im)) == 255 * im.size[0] * im.size[1]:
            self.label.configure(text="Пожалуйста, нарисуйте цифру.")
            return

        digit, acc = predict_digit(im)

        # Отладочные сообщения
        print(f"Предсказанная цифра: {digit}, Достоверность: {acc}")

        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')


    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
app.mainloop()
