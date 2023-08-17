from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

def read_file(path):
    df = pd.read_csv(path, header=None, sep="\t")
    df[1] = np.log1p(df[1].clip(lower=1e-5))
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, scaler

path = sys.argv[1]

df, _ = read_file(path)
model = load_model(os.path.join(os.path.dirname(__file__), 'model_ml.keras'))
pred = model.predict(np.expand_dims(df.values, axis=0))

if pred <= 0.5:
    print("Halal dengan probabilitas", round((1-pred[0][0])*100, 2), '%') 
else:
    print("Halal dengan probabilitas", round(pred[0][0]*100, 2), '%') 
