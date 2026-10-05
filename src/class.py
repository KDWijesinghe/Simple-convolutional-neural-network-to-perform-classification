from tensorflow.keras.datasets import fashion_mnist

from support.cnn import CNN

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = CNN(input_shape=(28, 28), num_classes=10)

model.fit(
    x_train,
    y_train,
    learning_rate=0.1,
    max_epochs=100,
    batch_size=256,
    verbose=True,
)

train_result = model.evaluate(x_train, y_train)
test_result = model.evaluate(x_test, y_test)

print(f"Train accuracy: {train_result['accuracy']:.2%}")
print(f"Test accuracy: {test_result['accuracy']:.2%}")
