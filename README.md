# Email Spam Detection System

![Python](https://img.shields.io/badge/python-3.9-blue) 
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange) 
![React](https://img.shields.io/badge/react-18-blue) 
![Flask](https://img.shields.io/badge/flask-2.x-lightgrey)

An end-to-end email spam detection system powered by a TensorFlow LSTM neural network with a modern web interface.

---

## 🚀 Features

- **Deep Learning Model**: LSTM-based neural network to classify emails as spam or ham
- **Modern Web UI**: Responsive React interface with animated result displays
- **REST API**: Flask-powered endpoints for integration and testing
- **Versatile Deployment**: Compatible with Docker, Heroku, Render, Vercel, and Railway
- **Fallback/Demo Mode**: Operates even when the trained model isn't present
  
---

## 🧠 System Architecture

```

Frontend (HTML) → API (Flask) → LSTM Model (TensorFlow)

````

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- Node.js (for the React frontend)

### 1. Clone the Repository

```bash
git clone https://github.com/anishgitflow/email-classification-fullstack.git
cd email-classification-fullstack
````

### 2. Set up the Backend

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train-model.py     # Train the LSTM model
```

### 3. Start the Flask API

```bash
python app.py
```

### 4. Set up the React Frontend

```bash
cd frontend
npm install
npm start
```

### Or Use the HTML Version

Simply open `index.html` in your browser.

---

## ☁️ Deployment Options

### Docker Compose

```bash
docker-compose up --build
```

## 🔌 API Endpoints

| Method | Endpoint      | Description            |
| ------ | ------------- | ---------------------- |
| POST   | `/predict`    | Analyze email content  |
| GET    | `/health`     | System health status   |
| GET    | `/model-info` | Returns model metadata |

---

## 📁 Project Structure

```
email-classification-fullstack/
├── app.py                # Flask API
├── train-model.py        # LSTM model training script
├── Dockerfile            # Docker image config
├── compose.yml           # Docker Compose config
├── requirements.txt      # Python dependencies
├── frontend/             # React frontend
│   ├── src/
│   └── package.json
├── index.html            # Optional HTML frontend
├── models/               # Saved model files
└── README.md             # Project documentation
```
## 📩 Demo

[Click Here](https://anishgitflow.github.io/email-classification-fullstack/)

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss any major changes.
