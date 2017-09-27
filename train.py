from keras import applications
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def train(X_train, y_train, X_validation, y_validation):

    # get the vgg model
    vgg = applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # add the dense layers to the base vgg model
    x = vgg.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=vgg.input, outputs=predictions)

    # train only the top layers
    for layer in vgg.layers:
        layer.trainable = False

    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=X_train,
              y=y_train,
              epochs=3,
              validation_data=(X_validation, y_validation))

    model.save('smoke_detection.h5')
    print('Model saved.')
    return
