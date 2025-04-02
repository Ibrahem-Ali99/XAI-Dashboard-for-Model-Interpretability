from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)





if __name__ == '__main__':
    app.run(debug=True)
