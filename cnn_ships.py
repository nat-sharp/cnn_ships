import pandas
import numpy
import cv2
import sklearn.model_selection
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from tensorflow import keras

def import_data(fileName):
    correct_pairs = pandas.read_csv(fileName+'.csv')

    images = numpy.array(correct_pairs['image'])
    categories = numpy.array(correct_pairs['category'])

    resized_images = []
    for image in images:
        current_image = cv2.imread('D:/6semestar/ORI/MojProj/cnn_ships/train/images/' + image, 1)
        result = cv2.resize(current_image, (210, 126))
        resized_images.append(result)

    resized_images = numpy.array(resized_images)

    return resized_images, categories


def main():
    images, categories = import_data('D:/6semestar/ORI/MojProj/cnn_ships/train/train')

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(images, categories, test_size=0.1)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    print('y_train shape after to_categorical:', y_train.shape)
    print('y_test shape after to_categorical:', y_test.shape)

    x_train = x_train/255.0
    x_test = x_test/255.0

    if keras.backend.image_data_format() == 'channels_first':
        input_shape = (None, 3, 126, 210)
    else:
        input_shape = (None, 126, 210, 3)

    model = keras.Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.build(input_shape)
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='rmsprop', #Adadelta()
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test))
    #model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))


    model.save('D:/6semestar/ORI/MojProj/cnn_ships/saved_model/SavedModel')


    #model = keras.models.load_model('path/to/location')

    #images, categories = import_data('D:/6semestar/ORI/MojProj/cnn_ships/sample_submission_ns2btKE')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



if __name__ == '__main__':
    main()



