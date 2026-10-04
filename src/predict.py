import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import argparse

# Input the path image 
parser = argparse.ArgumentParser(description="Predict a single character from an image.")
parser.add_argument("--image", required=True, help="Path to the input image file.")
args = parser.parse_args()

# Load model
print("Loading model...")
model = tf.keras.models.load_model('emnist_cnn_62_class_v1.h5')
print("Model loaded.")

label_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Preprocess input image in the same way EMNIST images were preprocessed
def preprocess_image(image_path):
    img = Image.open(image_path)

    # Convert to grayscale, flip, invert colors, resize
    img = img.convert('L')
    img = img.transpose(Image.Transpose.TRANSPOSE)
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    # Convert to NumPy array and normalize
    img_array = np.array(img) / 255.0

    # Reshape the data for the model (batch_size, height, width, channels) -> (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

def main():
    image_path = args.image
    processed_image = preprocess_image(image_path)

    if processed_image is not None:
        prediction = model.predict(processed_image)

        # Choose the highest probability from the array
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Map the index to the character
        predicted_character = label_mapping[predicted_index]

        print("\n--- Prediction Result ---")
        print(f"Predicted Character: {predicted_character}")
        print(f"Confidence: {confidence:.2%}")
        print("-------------------------\n")

if __name__ == "__main__":
    main()