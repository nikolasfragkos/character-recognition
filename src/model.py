def create_cnn_model(num_classes):
    return tf.keras.Sequential([
        # Data augmentation block
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.RandomRotation(0.1), # Random rotations
        tf.keras.layers.RandomZoom(0.1),     # Random zooms
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Random shifts

        # Block 1
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),

        # Block 2
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),

        # Block 3
        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),

        # Flatten and Classify
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])