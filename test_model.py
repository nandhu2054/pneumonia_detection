import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("pneumonia_model.h5")
print("✅ Model loaded successfully!")

# Function to test a single X-ray
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print(f"PNEUMONIA")
    else:
        print(f"NORMAL")

# Test your image
predict_image("C:\\Users\\S.D.N\\OneDrive\\Pictures\\pneumonia_detection\\pneumonia1.jpg")

