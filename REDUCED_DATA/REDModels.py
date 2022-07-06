import pandas as pd
import pickle
import os

# Load new data
clean = pd.read_csv('/Users/luispsalazar/Desktop/Team_3_Final_Project/REDUCED_DATA/OneNewRecord.csv')

# Define features set
new = clean.copy()

from sklearn.preprocessing import StandardScaler

# Creating StandardScaler instance
scaler = StandardScaler()

# Fitting Standard Scaler
X_scaler = scaler.fit(new)

# Scaling data
X_train_scaled = X_scaler.transform(new)

from sklearn.ensemble import GradientBoostingClassifier

# Making predictions with the saved & loaded model
filename = '/Users/luispsalazar/Desktop/Team_3_Final_Project/REDUCED_DATA/model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

predictions = loaded_model.predict(X_train_scaled)
pd.DataFrame({"Prediction": predictions})
print(predictions)