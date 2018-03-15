import pandas as pd
from geopy.distance import vincenty
# import datetime as dt
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
'''
A possible approach would be to sort the dataframe w.r.t close_date so that time leakage is handled by using
only the elements above to calculate nearest neighbour weights.
'''
# df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv').sort_values(by=['close_date'],kind='mergesort')
'''
Need a function to calculate the distance based on the coordinates.
'''
#y = df['close_price']
#X = df.drop(columns=['close_price'])
# print(df.head())
# print(df.tail())
# def time_leakage(X):


	# pass
# print(X.head())
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# knn_regressor = KNeighborsRegressor()
# knn_regressor.fit(X_train, y_train)
