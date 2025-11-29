# 🏥 Diabetes Prediction Web App

A machine learning-based web application that predicts the likelihood of diabetes based on health metrics using a Random Forest classifier.

## Features

- 🤖 **ML Model**: Random Forest classifier trained on diabetes dataset
- 🎨 **Modern UI**: Beautiful, responsive web interface with gradient design
- 📊 **8 Health Metrics**: Glucose, Blood Pressure, BMI, Age, and more
- ⚡ **Fast Predictions**: Real-time prediction results
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
diabetes-prediction/
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── data/
│   └── diabetes.csv               # Training dataset (768 samples)
├── models/
│   ├── model.joblib               # Trained Random Forest model
│   └── scaler.joblib              # StandardScaler for feature normalization
├── src/
│   ├── train_model.py             # Model training script
│   ├── data_preprocessing.py       # Data preprocessing pipeline
│   ├── predict.py                 # Prediction module
│   ├── utils.py                   # Utility functions
│   └── __init__.py
└── app/
    ├── templates/
    │   ├── index.html             # Input form page
    │   └── result.html            # Prediction result page
    └── static/
        └── style.css              # CSS styling
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python -m src.train_model --data_path data/diabetes.csv --model_dir models
```

### Running the Web App

```bash
python app.py
```

Then open your browser and navigate to: `http://localhost:5000`

## Input Features

- **Pregnancies**: Number of pregnancies (0-20)
- **Glucose**: Plasma glucose concentration (0-300 mg/dL)
- **Blood Pressure**: Diastolic blood pressure (0-200 mmHg)
- **Skin Thickness**: Triceps skin fold thickness (0-100 mm)
- **Insulin**: 2-Hour serum insulin (0-900 mIU/L)
- **BMI**: Body Mass Index (10-60 kg/m²)
- **Diabetes Pedigree Function**: Genetic risk factor (0-3)
- **Age**: Age in years (1-120)

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 8 health metrics
- **Training/Test Split**: 80/20
- **Feature Scaling**: StandardScaler normalization
- **Missing Values**: Zero values replaced with median for specific columns

## Disclaimer

⚠️ **Important**: This application is for educational purposes and should NOT be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for accurate diagnosis and treatment.

## Technologies Used

- **Backend**: Flask (Python web framework)
- **ML**: scikit-learn (Random Forest)
- **Frontend**: HTML5, CSS3
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

## License

This project is open source and available under the MIT License.

## Author

Created as a machine learning demonstration project.

---

**Note**: To use this app, you need to have trained models in the `models/` directory. Run the training script first if models don't exist.
