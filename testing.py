from tracemalloc import stop
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

# for PCA transformation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


# Remove nrows when ready for full dataset
# Remove 2K from name in last block for complete file
# WARNING, FILE "mergefiles.csv" IS 1.37 GB

delay_data = pd.read_csv('Resources/for_test.csv', encoding = 'utf-8')

#delay_data = pd.read_csv('mergefiles.csv', encoding = 'utf-8')
delay = pd.DataFrame(delay_data)
delay.head()

stop

# 1. Data Exploration
# Data type
delay.dtypes
delay.drop_duplicates()
delay.head()

# Missing data
delay.isna().sum()

# Number of unique values, and range
for col in delay.columns:
    print(f'"{col}" has {delay[col].nunique()} unique values, from {delay[col].min()} to {delay[col].max()}')

# Number of flights not delayed (0), and delayed (1)
delay['DEP_DEL15'].value_counts()

# Number of airports
delay['DEPARTING_AIRPORT'].nunique()

# Number of flights per airport
delay.groupby('DEPARTING_AIRPORT')['DEP_DEL15'].count()

# Replace " " with "_"
#delay = delay.replace(" ", "_", regex = True)
delay['PREVIOUS_AIRPORT'] = delay['PREVIOUS_AIRPORT'].str.replace(" ", "_")
delay['PREVIOUS_AIRPORT'] = delay['PREVIOUS_AIRPORT'].str.replace("/", "_")
delay['DEP_TIME_BLK'] = delay['DEP_TIME_BLK'].str.replace("-", "_")
delay['DEPARTING_AIRPORT'] = delay['DEPARTING_AIRPORT'].str.replace(" ", "_")
delay['DEPARTING_AIRPORT'] = delay['DEPARTING_AIRPORT'].str.replace("/", "_")
delay['DEPARTING_AIRPORT'] = delay['DEPARTING_AIRPORT'].str.replace(".", "")
delay['DEPARTING_AIRPORT'] = delay['DEPARTING_AIRPORT'].str.replace("'Hare", "Hare")
delay['DEPARTING_AIRPORT'] = delay['DEPARTING_AIRPORT'].str.replace("-", "_")
delay['CARRIER_NAME'] = delay['CARRIER_NAME'].str.replace(" ", "_")
delay['CARRIER_NAME'] = delay['CARRIER_NAME'].str.replace("/", "_")
delay['CARRIER_NAME'] = delay['CARRIER_NAME'].str.replace(".", "")
delay.head(5)

# Airport with least number of flights
delay.groupby('DEPARTING_AIRPORT')['DEP_DEL15'].count().min()

# Grouped by airport: count of delay/on-time
print(delay.groupby(['DEPARTING_AIRPORT','DEP_DEL15'])['MONTH'].agg('count'))

# Distance group: 1 to 11
# Distance group to be flown by departing aircraft.
sns.set_style('whitegrid')
delay["DISTANCE_GROUP"].plot(figsize = (16, 6))
delay['DISTANCE_GROUP'].plot(kind='hist', figsize = (12, 4))

# Segment number: 1 to 15
# The segment that this tail number is on for the day.
delay["SEGMENT_NUMBER"].plot(figsize = (16, 6))

delay['SEGMENT_NUMBER'].plot(kind = 'hist', figsize = (12, 4))

# Concurrent flights
delay["CONCURRENT_FLIGHTS"].plot(figsize = (16, 6))

delay['CONCURRENT_FLIGHTS'].plot(kind = 'hist', figsize = (12, 4))

# Number of seats
delay["NUMBER_OF_SEATS"].plot(figsize = (16, 6))

delay['NUMBER_OF_SEATS'].plot(kind = 'hist', figsize = (12, 4))

# Airport flights per month
delay["AIRPORT_FLIGHTS_MONTH"].plot(figsize = (16, 6))

delay['AIRPORT_FLIGHTS_MONTH'].plot(kind = 'hist', figsize = (12, 4))

# Airline flights per month
delay["AIRLINE_FLIGHTS_MONTH"].plot(figsize = (16, 6))

delay['AIRLINE_FLIGHTS_MONTH'].plot(kind = 'hist', figsize = (12, 4))

# AIRLINE_AIRPORT_FLIGHTS_MONTH
delay["AIRLINE_AIRPORT_FLIGHTS_MONTH"].plot(figsize = (16, 6))

delay['AIRLINE_AIRPORT_FLIGHTS_MONTH'].plot(kind = 'hist', figsize = (12, 4))

# AVG_MONTHLY_PASS_AIRPORT
delay["AVG_MONTHLY_PASS_AIRPORT"].plot(figsize = (16, 6))

delay['AVG_MONTHLY_PASS_AIRPORT'].plot(kind = 'hist', figsize = (12, 4))

# AVG_MONTHLY_PASS_AIRLINE
delay["AVG_MONTHLY_PASS_AIRLINE"].plot(figsize = (16, 6))

delay['AVG_MONTHLY_PASS_AIRLINE'].plot(kind = 'hist', figsize = (12, 4))

# FLT_ATTENDANTS_PER_PASS
delay["FLT_ATTENDANTS_PER_PASS"].plot(figsize = (16, 6))

delay['FLT_ATTENDANTS_PER_PASS'].plot(kind = 'hist', figsize = (12, 4))

# GROUND_SERV_PER_PASS
delay["GROUND_SERV_PER_PASS"].plot(figsize = (16, 6))

delay['GROUND_SERV_PER_PASS'].plot(kind = 'hist', figsize = (12, 4))

# Plane age
delay["PLANE_AGE"].plot(figsize = (16, 6))

delay['PLANE_AGE'].plot(kind = 'hist', figsize = (12, 4))

# Departing airport
delay.groupby('DEPARTING_AIRPORT')['DEP_DEL15'].count().plot(figsize = (16, 6))

# Previous airport
delay.groupby('PREVIOUS_AIRPORT')['DEP_DEL15'].count().plot(figsize = (16, 6))

# Precipitation
delay["PRCP"].plot(figsize = (16, 6))

delay['PRCP'].plot(kind = 'hist', figsize = (12, 4))

# Snow (SNOW): 0 to 17.2
# Inches of snowfall for day.
delay["SNOW"].plot(figsize = (16, 6))

delay['SNOW'].plot(kind = 'hist', figsize = (12, 4))

# SNOWD: 0 to 25.2
# Inches of snow on ground for day.
delay["SNWD"].plot(figsize = (16, 6))

delay['SNWD'].plot(kind='hist', figsize = (12, 4))

# Temperature
delay["TMAX"].plot(figsize = (16, 6))

delay['TMAX'].plot(kind = 'hist', figsize = (12, 4))

# Air wind speed (AWND): 0 to 33.78
# Max wind speed for day.
delay["AWND"].plot(figsize = (16, 6))

delay['AWND'].plot(kind = 'hist', figsize = (12, 4))

# 2. Data Cleanup
# DEP_TIME_BLK categorical variable

# Visualize the value counts
DEP_TIME_BLK_counts = delay.DEP_TIME_BLK.value_counts()
DEP_TIME_BLK_counts
DEP_TIME_BLK_counts.plot.density()

# Create the OneHotEncoder instance
enc = OneHotEncoder(sparse = False)

# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(delay.DEP_TIME_BLK.values.reshape(-1,1)))

# Rename encoded columns
encode_df.columns = enc.get_feature_names(['DEP_TIME_BLK'])
encode_df.head()

# Merge the two DataFrames together and drop the encoded column
delay_block = delay.merge(encode_df,left_index = True, right_index = True).drop(["DEP_TIME_BLK"], axis = 1)
delay_block.sample(5)

# Fit the encoder and produce encoded DataFrame (CARRIER_NAME)
encode_df = pd.DataFrame(enc.fit_transform(delay_block.CARRIER_NAME.values.reshape(-1,1)))

# Rename encoded columns
encode_df.columns = enc.get_feature_names(['CARRIER_NAME'])
encode_df.head()

# Merge the two DataFrames together and drop the encoded column
delay_carrier = delay_block.merge(encode_df,left_index = True, right_index = True).drop(["CARRIER_NAME"], axis = 1)
delay_carrier.sample(5)

# Convert DEPARTING_AIRPORT categorical variable
encode_df = pd.DataFrame(enc.fit_transform(delay_carrier.DEPARTING_AIRPORT.values.reshape(-1,1)))

# Rename encoded columns
encode_df.columns = enc.get_feature_names(['DEPARTING_AIRPORT'])
encode_df.head()

# Merge the two DataFrames together and drop the encoded column
delay_departing = delay_carrier.merge(encode_df,left_index = True, right_index = True).drop(["DEPARTING_AIRPORT"], axis = 1)
delay_departing.sample(5)

# Replace with 1 if a previous airport exists (connection flight)
delay_departing["PREVIOUS_AIRPORT"] = np.where(delay_departing["PREVIOUS_AIRPORT"] == "NONE", 0, 1)

# Apply PCA to airport coordinates, reduce from two to one feature
coord_pca = delay_departing[['LATITUDE', 'LONGITUDE']]
coord_pca.head()

# Initialize PCA model
pca = PCA(n_components = 1)

# Get principal component for the dataset
transfor_coord = pca.fit_transform(coord_pca)
transfor_coord

# Transform PCA data to a DataFrame
new_coord = pd.DataFrame(data = transfor_coord, columns = ["principal_component"])
new_coord.sample(5)

# Explained variance
pca.explained_variance_ratio_

# Remove LATITUDE and LONGITUD
# Merge dataframe with one PCA feature
delay_departing.drop(['LATITUDE', 'LONGITUDE'], axis = 1, inplace = True)

delay_withPCA = pd.concat([delay_departing, new_coord], axis = 1)
delay_withPCA

delay_withPCA = delay_withPCA[['DEP_DEL15'] + [col for col in delay_withPCA.columns if col != 'DEP_DEL15']]
delay_withPCA

for i in delay_withPCA.columns:
    print(i)

# Save cleaned data as a new csv file
#      ADD NEWNAME to GITIGNORE!!!!!
delay_withPCA.to_csv("./Resources/delay_clean.csv", index = False)