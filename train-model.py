import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SpamDetectionTrainer:
    def __init__(self, max_features=10000, max_length=100, embedding_dim=128):
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        
    def load_data(self, file_path='spam.csv'):
        """Load and prepare the spam dataset"""
        try:
            # Try to load the dataset
            df = pd.read_csv(file_path, encoding='latin-1')
            
            # Assuming standard spam dataset format with columns 'v1' (label) and 'v2' (message)
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df[['v1', 'v2']].copy()
                df.columns = ['label', 'message']
            elif 'Category' in df.columns and 'Message' in df.columns:
                df = df[['Category', 'Message']].copy()
                df.columns = ['label', 'message']
            else:
                raise ValueError("Dataset should have label and message columns")
                
            # Convert labels to binary (0 for ham, 1 for spam)
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            print(f"Dataset loaded successfully: {len(df)} samples")
            print(f"Spam: {sum(df['label'])}, Ham: {len(df) - sum(df['label'])}")
            
            return df
            
        except FileNotFoundError:
            print("Dataset file not found. Creating synthetic dataset for demo...")
            return self.create_synthetic_dataset()
    
    def create_synthetic_dataset(self):
        """Create a synthetic dataset for demonstration"""
        spam_messages = [
            "WINNER! You have won $1000000! Call now to claim your prize!",
            "FREE! Get your free iPhone now! Limited time offer!",
            "Congratulations! You are selected for cash prize of $500000",
            "URGENT! Your account will be closed. Click here immediately",
            "Amazing weight loss pills! Lose 30kg in 30 days guaranteed!",
            "Hot singles in your area! Click to meet them now!",
            "Make money from home! $5000 per week working part time!",
            "LOTTERY WINNER! Claim your prize before it expires!",
            "FREE MONEY! No strings attached! Click here now!",
            "Discount 90% OFF! Buy now while stocks last!",
            "URGENT RESPONSE REQUIRED! Send your details immediately",
            "You've been pre-approved for a $50000 loan!",
            "Get rich quick! Investment opportunity of lifetime!",
            "Free vacation to Bahamas! You are our lucky winner!",
            "ALERT! Suspicious activity on your account! Verify now!",
        ] * 20  # Repeat to create more samples
        
        ham_messages = [
            "Hi John, can we schedule a meeting for tomorrow at 2 PM?",
            "Please review the attached document and let me know your thoughts",
            "Thank you for your presentation yesterday. It was very informative",
            "The project deadline has been moved to next Friday",
            "Don't forget about the team lunch tomorrow at noon",
            "I've sent you the updated spreadsheet with the latest figures",
            "Could you please provide an update on the marketing campaign?",
            "The conference call is scheduled for 3 PM today",
            "Please find the meeting minutes attached to this email",
            "Your invoice has been processed and payment is on the way",
            "Welcome to our team! We're excited to have you aboard",
            "The server maintenance will be conducted this weekend",
            "Please submit your expense reports by end of this week",
            "Happy birthday! Hope you have a wonderful day",
            "The quarterly report is due next Monday",
        ] * 20  # Repeat to create more samples
        
        # Create DataFrame
        messages = spam_messages + ham_messages
        labels = [1] * len(spam_messages) + [0] * len(ham_messages)
        
        df = pd.DataFrame({
            'message': messages,
            'label': labels
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        print(f"Synthetic dataset created: {len(df)} samples")
        print(f"Spam: {sum(df['label'])}, Ham: {len(df) - sum(df['label'])}")
        
        return df
    
    def preprocess_text(self, text):
        """Preprocess text data"""
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
    
    def prepare_data(self, df):
        """Prepare text data for training"""
        # Preprocess messages
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['processed_message'])
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(df['processed_message'])
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length)
        y = df['label'].values
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Sequence shape: {X.shape}")
        
        return X, y
    
    def build_model(self):
        """Build LSTM model for spam detection"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=self.max_features,
                output_dim=self.embedding_dim,
                input_length=self.max_length
            ),
            tf.keras.layers.SpatialDropout1D(0.3),
            tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train_model(self, X, y, test_size=0.2, validation_split=0.1, epochs=10, batch_size=32):
        """Train the LSTM model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
        
        # Predictions for detailed evaluation
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return history, (X_test, y_test)
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_tokenizer(self):
        """Save the trained model and tokenizer"""
        # Save model
        self.model.save('spam_model.h5')
        print("Model saved as 'spam_model.h5'")
        
        # Save tokenizer
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer saved as 'tokenizer.pickle'")
        
        # Save model architecture summary
        with open('model_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print("Model summary saved as 'model_summary.txt'")
    
    def test_predictions(self, test_texts=None):
        """Test model predictions on sample texts"""
        if test_texts is None:
            test_texts = [
                "WINNER! You have won $1000000! Call now!",
                "Hi John, can we schedule a meeting tomorrow?",
                "FREE MONEY! Click here now!",
                "Please review the attached document",
                "URGENT! Your account will be closed immediately!"
            ]
        
        print("\nTesting predictions:")
        print("-" * 50)
        
        for text in test_texts:
            # Preprocess
            processed = self.preprocess_text(text)
            sequence = self.tokenizer.texts_to_sequences([processed])
            padded = pad_sequences(sequence, maxlen=self.max_length)
            
            # Predict
            prediction = self.model.predict(padded, verbose=0)[0][0]
            label = "SPAM" if prediction > 0.5 else "HAM"
            confidence = max(prediction, 1 - prediction)
            
            print(f"Text: {text[:50]}...")
            print(f"Prediction: {label} (Confidence: {confidence:.3f})")
            print(f"Spam Probability: {prediction:.3f}")
            print("-" * 50)

def main():
    """Main training function"""
    print("Starting Spam Detection Model Training...")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SpamDetectionTrainer(
        max_features=10000,
        max_length=100,
        embedding_dim=128
    )
    
    # Load data
    df = trainer.load_data()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Build model
    model = trainer.build_model()
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    history, test_data = trainer.train_model(X, y, epochs=15)
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Test predictions
    trainer.test_predictions()
    
    # Save model and tokenizer
    trainer.save_model_and_tokenizer()
    
    print("\nTraining completed successfully!")
    print("Files saved:")
    print("- spam_model.h5 (trained model)")
    print("- tokenizer.pickle (tokenizer)")
    print("- model_summary.txt (model architecture)")
    print("- training_history.png (training plots)")
    print("- confusion_matrix.png (evaluation plot)")

if __name__ == "__main__":
    main()