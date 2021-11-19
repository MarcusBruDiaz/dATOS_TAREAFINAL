import json
import numpy as np
import os
import pickle
import joblib
import pandas as pd
def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'proyecc_model.pkl')
    model = joblib.load(model_path)
def run(raw_data):

    data =json.loads(raw_data)
 #   df = pd.DataFrame()
 #   df['Day']=data[0][0]
 #   df['Holiday']=data[0][1]
 #   df['P_H_21']=data[0][2]
  #  df['P_H_22']=data[0][3]
  #  df['P_H_23']=data[0][4]
  #  df['P_H_24']=data[0][5] 
    # Make prediction.
    inp=pd.DataFrame(data,index=[0])
    y_hat = model.predict(inp)
    return json.dumps(y_hat[0][0])

