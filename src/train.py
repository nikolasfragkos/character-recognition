import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from model import create_cnn_model

# Loading the EMNIST dataset and preprocessing the data
def load_and_prepare_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/byclass',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    # Set number of classes and label mapping according to the byclass dataset
    num_classes = ds_info.features['label'].num_classes
    label_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def preprocess_byclass(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        return image, label

    batch_size = 128

    ds_train = ds_train.map(preprocess_byclass).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.map(preprocess_byclass).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, num_classes, label_mapping

def main():
    #Load and preprocess data
    ds_train, ds_test, num_classes, label_mapping = load_and_prepare_data()
    
    # Create and compile the model
    model = create_cnn_model(num_classes)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # Callbacks for smarter training
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=0.00001)

    print("\nTraining on EMNIST Dataset...")
    history = model.fit(
        ds_train,
        epochs=25,
        validation_data=ds_test,
        callbacks=[early_stopping, reduce_lr]
    )

    model.save('emnist_cnn_62_class_v3.h5')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()