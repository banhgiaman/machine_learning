import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPool2D


def model_fn(
    input_shape=(28,28,1), num_class=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(20, 5, activation='tanh', padding='same')(inputs)
    x = MaxPool2D()(x)
    x = Conv2D(50, 5, activation='tanh', padding='same')(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(500)(x)
    outputs = Dense(num_class)(x)
    model = Model(inputs=inputs, outputs=outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
            
if __name__ == '__main__':
    import os
    import glob
    import numpy as np
    
    def get_generator(images, labels):
        for idx in range(len(images)):
            yield images[idx].reshape(28,28,1), labels[idx]
    
    
    epochs = 5
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    model = model_fn()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('..', 'models', 'lenet', 'checkpoints'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    # model.summary()
    
    train_ds = tf.data.Dataset.from_generator(
        get_generator,
        args=(train_images[:int(len(train_images)*0.8)], train_labels[:int(len(train_labels)*0.8)]),
        output_types=(tf.float64, np.uint8),
        output_shapes=((28, 28, 1), ())
    ).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_generator(
        get_generator,
        args=(train_images[int(len(train_images)*0.8):], train_labels[int(len(train_labels)*0.8):]),
        output_types=(tf.float64, np.uint8),
        output_shapes=((28, 28, 1), ())
    ).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_generator(
        get_generator,
        args=(test_images, test_labels),
        output_types=(tf.float64, np.uint8),
        output_shapes=((28, 28, 1), ())
    ).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    # model.save(os.path.join('..', 'models', 'lenet'))
    
    