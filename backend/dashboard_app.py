from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os
import uuid
import logging

app = Flask(__name__)
CORS(app)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "XGBoost": XGBRegressor
}

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        X = pd.DataFrame(data['X'])
        y = pd.Series(data['y'])
        task = data['task']
        model_name = data['model']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Validate regression targets
        if task == "Regression":
            if not pd.api.types.is_numeric_dtype(y):
                return jsonify({
                    "status": "error",
                    "message": "Regression requires numerical target variable. "
                               f"Found non-numerical values: {list(y.unique())}"
                }), 400
        
        # Preprocessing for features
        numeric_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Create pipeline with target encoding for classification
        if task == "Classification":
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', CLASSIFICATION_MODELS[model_name]())
            ])
        else:
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', REGRESSION_MODELS[model_name]())
            ])

        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if task == "Classification":
            metrics["accuracy"] = accuracy_score(y_test, predictions)
            metrics["f1_score"] = f1_score(y_test, predictions, average='weighted')
            metrics["precision"] = precision_score(y_test, predictions, average='weighted')
            metrics["recall"] = recall_score(y_test, predictions, average='weighted')
        else:
            metrics["mse"] = mean_squared_error(y_test, predictions)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2"] = r2_score(y_test, predictions)
        
        # Save artifacts
        model_id = str(uuid.uuid4())
        artifacts = {
            'pipeline': model,
            'label_encoder': label_encoder if task == "Classification" else None
        }
        joblib.dump(artifacts, f"{MODEL_DIR}/{model_id}.joblib")
        
        return jsonify({
            "status": "success",
            "model_id": model_id,
            "metrics": metrics
        })
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)