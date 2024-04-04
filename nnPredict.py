import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from Convert import convert_png_to_mnist
import os.path


modelVal = './NNvalues.h5'


(X_train,y_train),(X_test,y_test) = mnist.load_data()

print(X_train.shape)
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

X_train = X_train/255
X_test = X_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes= y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def prepare_NN():
    model = baseline_model()
    if os.path.isfile(modelVal):
        model.load_weights(modelVal)
    else:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)
        model.save_weights(modelVal)
    scores = model.evaluate(X_test,y_test,verbose=0)
    print('Err: %.2f%%'%(100 - scores[1] * 100))
    return model

def Predict(model,imageFile):
    X_gray=convert_png_to_mnist(imageFile)
    X_gray=X_gray.reshape(1,num_pixels).astype('float32')
    prediction_NN = model.predict(np.array([X_gray[0]]))
    prediction = 0
    for i in range(0,10):
        if prediction_NN[0][i]==1 :
            print(i)
            prediction = i
    return prediction
    # print(prediction)


# model = prepare_NN()
