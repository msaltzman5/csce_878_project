import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

def main():

  all_rides = pd.read_csv('data/cab_rides.csv')

  all_rides = all_rides[['distance', 'price', 'cab_type', '']].dropna()


  # filtered_rides = all_rides[['price', 'beds', 'baths', 'sqft']].dropna()
  all_rides.head()

  X = all_rides[['distance']]
  y = all_rides['price']

  # Scale the input features
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  eNet = ElasticNet(alpha=.1, l1_ratio=.5)
  eNet.fit(X_scaled,y)

  yPredictedENet = eNet.predict(X_scaled)
  
  all_rides['predicted_price'] = yPredictedENet

  print(all_rides[['distance', 'price', 'predicted_price']])

  print("Done!")

if __name__ == "__main__":
    main()