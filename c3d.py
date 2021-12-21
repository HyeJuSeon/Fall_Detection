import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten, Dropout

def C3D():
    with tf.device('/device:GPU:0'):
        model = Sequential()
        input_shape = (16, 112, 112, 3) 

        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv1a', input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name='pool1'))

        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv2a', input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))

        model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv3a', input_shape=input_shape))
        model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv3b', input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))

        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv4a', input_shape=input_shape))
        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv4b', input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))

        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv5a', input_shape=input_shape))
        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',padding='same', name='conv5b', input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))

        model.add(Flatten())

        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))

        model.add(Dense(2, activation='sigmoid'))
        model.summary()

        return model
      
