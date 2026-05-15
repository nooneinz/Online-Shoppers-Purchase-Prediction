# Online Shoppers Purchasing Intention Prediction

## 📋 Project Description

This project uses **Artificial Neural Networks (ANN)** to predict whether a website visitor will make a purchase or not.

### The Problem 🎯
In e-commerce stores, it is crucial to identify visitors who have genuine purchasing intentions to improve marketing and conversion strategies. This model helps in:
- Identifying potential buyers
- Improving the conversion rate (Conversion Rate)
- Personalizing offers and recommendations

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Building neural networks
- **Pandas** - Data processing
- **Scikit-learn** - Data splitting and statistical metrics
- **NumPy** - Mathematical operations
- **Matplotlib & Seaborn** - Data visualization

---

## 📊 Data Structure

The dataset includes information about:
- Types of pages visited
- Duration of visits
- Bounce Rate
- Exit Rate
- Month and day type (weekend or not)
- Visitor type

---

## 🚀 How to Run

### 1️⃣ Install Libraries
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Notebook
```bash
jupyter notebook ANN_Assignment.ipynb
```

### 3️⃣ Launch the Interactive Web App
After training the model, run the Streamlit application:
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser to use the interface.

### 4️⃣ Main Code Steps

#### Cell 1: Import Libraries
All required libraries from TensorFlow, Pandas, and others are imported.

#### Cell 2: Data Preparation
- Load CSV file
- Handle missing values
- Convert categorical variables to numbers
- Split data (80% training, 20% testing)
- Standardize data using StandardScaler

#### Cell 3: Build the Model
```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),      # Input layer
    Dense(32, activation='relu'),           # Hidden layer with 32 units
    Dense(16, activation='relu'),           # Hidden layer with 16 units
    Dense(1, activation='sigmoid')          # Output layer (binary classification)
])
```

**Handling Data Imbalance:**
```python
class_weights = {0: 1., 1: 3.}  # Give more weight to buyers (class 1)
```

#### Cell 4: Evaluation and Results
- Print Classification Report
- Plot Confusion Matrix
- Save the model as `online_shoppers_model.keras`
- Save the Scaler as `scaler.pkl` for future data preprocessing

#### Cell 5: Feature Importance Analysis
- Uses SHAP (SHapley Additive exPlanations) to explain model predictions
- Shows which features have the most impact on purchase decisions
- Visualizes feature importance with bar plots
- Helps understand the business drivers behind predictions

---

## 🚀 Advanced Features

### 1️⃣ Scaler Persistence
The StandardScaler is saved as `scaler.pkl` to ensure new customer data is preprocessed the same way as training data:
```python
import joblib
scaler = joblib.load('scaler.pkl')
new_data_scaled = scaler.transform(new_data)
```

### 2️⃣ Interactive Web Application
Run the Streamlit web app for real-time predictions:
```bash
streamlit run app.py
```

This launches an interactive interface where you can:
- Input visitor behavioral metrics
- Get instant purchase predictions
- See confidence scores
- Receive optimization recommendations

### 3️⃣ Feature Importance Visualization
Understand WHY the model makes decisions through SHAP analysis:
- Identify key factors influencing purchase behavior
- Support business decision-making
- Optimize website experience based on insights

---

## 📈 Expected Results

- **Model Accuracy:** ~88-89%
- **Improved Recall for Buyers:** Through class_weights
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives

---

## 💾 Using the Saved Model

After running the notebook, the model will be saved as `online_shoppers_model.keras`

To use it in a Backend API:
```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('online_shoppers_model.keras')

# Make predictions on new data
new_customer = np.array([[...]])  # New customer data
prediction = model.predict(new_customer)
```



## ⚠️ Important Notes

1. **Data Imbalance:** The model was trained with higher weight for class 1 (buyers) to improve Recall
2. **Scaling:** Data was standardized before training to improve performance
3. **Validation Split:** 20% of training data was used for validation during training

---

## 🌐 Deployment Options

### Deploy Web App on Streamlit Cloud (FREE)
1. Push your project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Streamlit automatically deploys and updates with each push
5. Share the link on LinkedIn with a demo video!

### Deploy Backend API on Heroku/Railway
Create a `main.py` Flask/FastAPI server to serve predictions:
```python
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
model = load_model('online_shoppers_model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    scaled = scaler.transform([data['features']])
    prediction = model.predict(scaled)
    return jsonify({'prediction': float(prediction[0][0])})

if __name__ == '__main__':
    app.run()
```

---

## 💼 Full Stack Integration

This project demonstrates complete ML + Web Development skills:
- **Backend**: Python, TensorFlow, Model serving
- **Frontend**: Streamlit interactive UI
- **Data Engineering**: Preprocessing, scaling, balancing
- **DevOps**: Containerization (Docker), Cloud deployment
- **Analysis**: Feature importance, model interpretability

---
 ## 👩‍💻 Author

**Sara Saleh Said Al-Harbi**
- **LinkedIn:** [www.linkedin.com/in/sara-alharbi-1b53263a2]
- **GitHub:** [https://github.com/nooneinz]

*Final-year university student specializing in AI and Machine Learning .*


