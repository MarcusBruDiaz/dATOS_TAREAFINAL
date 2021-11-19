import pickle
import numpy as np
import json
import pandas as pd
# Load the trained model from current directory
with open('./outputs/proyecc_model.pkl', 'rb') as model_pkl:
    lr = pickle.load(model_pkl)
if __name__ == "__main__":
    df = pd.DataFrame()
    df['Day']=[0]
    df['Holiday']=0
    df['P_H_21']=1162700
    df['P_H_22']=1105959
    df['P_H_23']=1028623
    df['P_H_24']=945418
    predict_result = lr.predict(df)
    
    print(predict_result)