import tensorflow as tf
from tensorflow import keras
import tkinter
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
import numpy as np
import cv2
import os

# define the name of the directory to be created
path = "/HandWrittenNumbers"

message = "No Current Guess"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# setting up handwritten digit data set
# data set will contain 28 by 28 greyscale images of the handwritten digits
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
Y_train = keras.utils.to_categorical(Y_train, n_classes)
Y_test = keras.utils.to_categorical(Y_test, n_classes)

# setting up the model
model = tf.keras.models.Sequential()

# input layer
model.add(keras.layers.Dense(512, input_shape=(784,)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.2))

# layer 2
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.2))

# output layer
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

# compiling model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, Y_train, epochs=5)
model.evaluate(X_test, Y_test)

previousX = -1
previousY = -1
root = tkinter.Tk()
root.title("Handwritten Number Recognition")
root.geometry('500x286')
w = tkinter.Canvas(root, width=282, height=282)
w.grid(column=0, row=0)
w.create_rectangle(4, 4, 282, 282, outline="black", width=2)

def getter(widget):
    root.update()
    x = widget.winfo_rootx() + widget.winfo_x()
    y = widget.winfo_rooty() + widget.winfo_y()
    ImageGrab.grab().crop((x+8, y+28, x+554, y+554)).save("/Users/ishansethi/Desktop/Quad 2/temp.png")

def resetDrawings():
    global image1, draw
    w.delete("draw")

def resetHeld(event):
    global previousX, previousY
    previousX = -1
    previousY = -1

def drag(event):
    global previousX, previousY
    if previousX != -1 and previousY != -1:
        w.create_line(previousX, previousY, event.x, event.y, width=21, tag="draw")
    if 4 < event.x < 282 and 4 < event.y < 282:
        w.create_oval(event.x - 10, event.y - 10, event.x + 10, event.y + 10, fill="black", tag="draw")
        previousX = event.x
        previousY = event.y
    else:
        previousX = -1
        previousY = -1
    return

def guess():
    global message
    getter(root)
    image = cv2.imread("/Users/ishansethi/Desktop/Quad 2/temp.png")
    res = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_LANCZOS4)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.bitwise_not(res)
    cv2.imwrite("/Users/ishansethi/Desktop/Quad 2/28vy28.png", res)
    data = np.asarray(res)
    data = data / 255.0
    data = data.flatten()
    data = np.asarray([data, data])
    predict = np.argmax(model.predict(data)[0])
    label["text"] = "Current Guess: " + str(predict)
    print(predict)


w.bind('<B1-Motion>', drag)
w.bind('<ButtonRelease-1>', resetHeld)
resetButton = tkinter.Button(root, text="Reset", command=resetDrawings)
resetButton.grid(column=1, row=0)
guessButton = tkinter.Button(root, text="Submit Guess", command=guess)
guessButton.grid(column=2, row=0)
label = tkinter.Label(root, text=message)
label.grid(column=3, row=0)
root.mainloop()
