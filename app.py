import xgboost as xgb
import pickle as pkl
import pandas as pd
import numpy as np
import sklearn
import joblib

from sklearn.metrics import mean_squared_error as MSE
from flask import Flask, request, jsonify
from sklearn.metrics import r2_score

# init Flask app
app = Flask(__name__)

# helper function for evaluating a model
def evaluate(labels, predictions):
    # compute for rmse values and r^2 scores 
    rmse = np.sqrt(MSE(labels, predictions))
    r2 = r2_score(labels, predictions)
    
    return rmse, r2

# home page
@app.route("/")
def index():
    return "Hello, World!"

# route which will accept POST requests and return our model predictions
@app.route('/predict', methods=['POST'])
def prediction():
    # load predictor models
    lr_model = joblib.load('models/capstone_lr_model.pkl')
    rf_model = joblib.load('models/capstone_rf_model.pkl')
    xgb_model = pkl.load(open('models/xgboost-model', 'rb'))
    
    # prepare input data
    content = request.json
    temp = np.array(content)
    payload = temp.reshape(-1, 7)

    # load a LIBSVM text file or a XGBoost binary file into DMatrix object 
    # (an internal data structure that is used by XGBoost, which is optimized for both memory efficiency and training speed)
    xgmat = xgb.DMatrix(payload)
    
    # perform predictions
    lr_predict = lr_model.predict(payload)
    rf_predict = rf_model.predict(payload)
    xgb_predict = xgb_model.predict(xgmat)
    
    target_vals = pd.read_csv('data/test_data.csv')
    
    lr_rmse, lr_accuracy = evaluate(target_vals['target'], lr_predict)
    rf_rmse, rf_accuracy = evaluate(target_vals['target'], rf_predict)
    xgb_rmse, xgb_accuracy = evaluate(target_vals['target'], xgb_predict)
    
    res = dict(
        {
            'lr_rmse': lr_rmse,
            'lr_accuracy': lr_accuracy * 100,
            'rf_rmse': rf_rmse,
            'rf_accuracy': rf_accuracy * 100,
            'xgb_rmse': xgb_rmse,
            'xgb_accuracy': xgb_accuracy * 100
        }
    )
    
    output_dict = dict(
        {
            'lr_predictions': lr_predict.tolist(), 
            'rf_predictions': rf_predict.tolist(), 
            'xgb_predictions': xgb_predict.tolist(), 
            'actual_values': target_vals['target'].values.tolist(),
            # verbose functionality is a WIP
            # 'results': res,
        }
    )

    return output_dict

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)