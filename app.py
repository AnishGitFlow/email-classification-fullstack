from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class SpamDetector:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = 100
        self.model_loaded = False
        
    def preprocess_text(self, text):
        """Preprocess text similar to training preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation and digits
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        try:
            # Try to load saved model
            if os.path.exists('spam_model.h5'):
                self.model = tf.keras.models.load_model('spam_model.h5')
                logger.info("Model loaded successfully from spam_model.h5")
            else:
                # Create a demo model if no trained model exists
                logger.warning("No saved model found. Creating demo model...")
                self.create_demo_model()
            
            # Load tokenizer
            if os.path.exists('tokenizer.pickle'):
                with open('tokenizer.pickle', 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                logger.info("Tokenizer loaded successfully")
            else:
                logger.warning("No saved tokenizer found. Creating demo tokenizer...")
                self.create_demo_tokenizer()
                
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.create_demo_model()
            self.create_demo_tokenizer()
            self.model_loaded = True
    
    def create_demo_model(self):
        """Create a demo LSTM model for testing purposes"""
        vocab_size = 10000
        embedding_dim = 128
        max_length = self.max_length
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Demo model created successfully")
    
    def create_demo_tokenizer(self):
        """Create a demo tokenizer"""
        # Common words for demonstration
        demo_texts = [
            "free money win lottery prize click now",
            "meeting project work schedule team update",
            "buy discount offer limited time urgent",
            "please review document thank you regards",
            "congratulations winner claim prize immediate",
            "hello how are you today meeting tomorrow"
        ]
        
        self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(demo_texts)
        logger.info("Demo tokenizer created successfully")
    
    def predict(self, text):
        """Predict if text is spam or ham"""
        if not self.model_loaded:
            raise Exception("Model not loaded")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize and pad
        sequences = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        
        # Predict
        prediction = self.model.predict(padded_sequences, verbose=0)[0][0]
        
        return {
            'spam_probability': float(prediction),
            'ham_probability': float(1 - prediction),
            'prediction': 'spam' if prediction > 0.5 else 'ham',
            'confidence': float(max(prediction, 1 - prediction))
        }

# Initialize spam detector
spam_detector = SpamDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': spam_detector.model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict_spam():
    """Main prediction endpoint"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        result = spam_detector.predict(text)
        
        logger.info(f"Prediction made: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if not spam_detector.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get model summary info
        total_params = spam_detector.model.count_params()
        
        return jsonify({
            'model_type': 'LSTM',
            'max_sequence_length': spam_detector.max_length,
            'total_parameters': int(total_params),
            'vocab_size': len(spam_detector.tokenizer.word_index) if spam_detector.tokenizer else 0
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.before_first_request
def initialize():
    """Initialize the model when the app starts"""
    logger.info("Initializing spam detector...")
    spam_detector.load_model_and_tokenizer()
    logger.info("Spam detector initialized successfully")

if __name__ == '__main__':
    # Load model and tokenizer
    spam_detector.load_model_and_tokenizer()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)