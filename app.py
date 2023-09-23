from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model here
# Example:
model = tf.keras.models.load_model('plant_identification_model.h5')

# Define your label mapping dictionary
# Example:
label_mapping = {0: 'Alpinia Galanga (Rasna)',
 1: 'Amaranthus Viridis (Arive-Dantu)',
 2: 'Artocarpus Heterophyllus (Jackfruit)',
 3: 'Azadirachta Indica (Neem)',
 4: 'Basella Alba (Basale)',
 5: 'Brassica Juncea (Indian Mustard)',
 6: 'Carissa Carandas (Karanda)',
 7: 'Citrus Limon (Lemon)',
 8: 'Ficus Auriculata (Roxburgh fig)',
 9: 'Ficus Religiosa (Peepal Tree)',
 10: 'Hibiscus Rosa-sinensis',
 11: 'Jasminum (Jasmine)',
 12: 'Mangifera Indica (Mango)',
 13: 'Mentha (Mint)',
 14: 'Moringa Oleifera (Drumstick)',
 15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
 16: 'Murraya Koenigii (Curry)',
 17: 'Nerium Oleander (Oleander)',
 18: 'Nyctanthes Arbor-tristis (Parijata)',
 19: 'Ocimum Tenuiflorum (Tulsi)',
 20: 'Piper Betle (Betel)',
 21: 'Plectranthus Amboinicus (Mexican Mint)',
 22: 'Pongamia Pinnata (Indian Beech)',
 23: 'Psidium Guajava (Guava)',
 24: 'Punica Granatum (Pomegranate)',
 25: 'Santalum Album (Sandalwood)',
 26: 'Syzygium Cumini (Jamun)',
 27: 'Syzygium Jambos (Rose Apple)',
 28: 'Tabernaemontana Divaricata (Crape Jasmine)',
 29: 'Trigonella Foenum-graecum (Fenugreek)'}

def preprocess_image(image):
    # Resize the image to match the input size of your model (e.g., 224x224)
    image = image.resize((224, 224))
    
    # Convert the image to an array and preprocess for your specific model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values (if required)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def process_predictions(predictions):
    # Assuming you have a label_mapping dictionary defined
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, 'Unknown')
    confidence = predictions[0][predicted_label_index]
    
    return f'Predicted Label: {predicted_label}, Confidence: {confidence:.2f}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file from the form
    uploaded_image = request.files['image']

    if uploaded_image.filename != '':
        # Open and preprocess the image
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)

        # Make predictions using your model
        predictions = model.predict(preprocessed_image)

        # Process the predictions and return the result
        result = process_predictions(predictions)
        return render_template('result.html', result=result)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
