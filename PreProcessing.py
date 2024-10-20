import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_data.csv')

# Create masks for each anomaly condition
temp_anomaly = (df['temp'] > 60) | (df['temp'] <= 0)
respiration_rate_anomaly = df['respirationRate'] <= 0
heart_rate_anomaly = df['heartRate'] <= 0

# Aggregate anomalies for each ID
anomalies = temp_anomaly | respiration_rate_anomaly | heart_rate_anomaly
anomaly_counts = anomalies.groupby(df['id']).sum()

# Sort IDs by the total number of anomalies
anomaly_counts = anomaly_counts.sort_values(ascending=True)
#print(anomaly_counts.head())

# Filter IDs with 119 anomalies or less
filtered_ids = anomaly_counts[anomaly_counts <= 119].index
filtered_df = df[df['id'].isin(filtered_ids)]
num_ids = filtered_df['id'].nunique()
print("Number of IDs:", num_ids)

# Create masks for each anomaly condition in the filtered dataset (20 ids)
temp_anomaly = (filtered_df['temp'] > 60) | (filtered_df['temp'] <= 0)
respiration_rate_anomaly = filtered_df['respirationRate'] <= 0
heart_rate_anomaly = filtered_df['heartRate'] <= 0

# Replace anomalies with NaNs
filtered_df.loc[temp_anomaly, 'temp'] = np.nan
filtered_df.loc[respiration_rate_anomaly, 'respirationRate'] = np.nan
filtered_df.loc[heart_rate_anomaly, 'heartRate'] = np.nan

filtered_df.loc[:, 'temp'] = filtered_df['temp'].interpolate(method='linear')
filtered_df.loc[:, 'respirationRate']= filtered_df['heartRate'].interpolate(method='linear')
filtered_df.loc[:, 'heartRate'] = filtered_df['respirationRate'].interpolate(method='linear')

# Check if there are any remaining NaNs after imputation
#print('is null ? ')
#print(filtered_df.isnull().sum())

temp_anomaly = (filtered_df['temp'] > 60) | (filtered_df['temp'] <= 0)
respiration_rate_anomaly = filtered_df['respirationRate'] <= 0
heart_rate_anomaly = filtered_df['heartRate'] <= 0

anomalies = temp_anomaly | respiration_rate_anomaly | heart_rate_anomaly
anomaly_counts = anomalies.groupby(filtered_df['id']).sum()

# Sort IDs by the total number of anomalies
anomaly_counts = anomaly_counts.sort_values(ascending=True)
print(anomaly_counts.head(20))
num_ids = filtered_df['id'].nunique()
print("Number of IDs:", num_ids)

filtered_df = filtered_df[filtered_df['id'] != 51]

df_grouped = filtered_df.groupby('id').agg({
    'gsr_x': list, 'altitude': list, 'peakAcceleration': list,
    'ibi': list, 'temp': list, 'x': list, 'y': list, 'z': list,
    'heartRate': list, 'respirationRate': list, 'heartRateVariability': list,
    'physicalTiredLevel': list
}).reset_index()

print(df_grouped.shape)

print(filtered_df.groupby('id')['id'].value_counts().sort_index())
num_ids = filtered_df['id'].nunique()
print("Number of IDs:", num_ids)



# Print the distribution of every class of physicalTiredLevel
print("Distribution of physicalTiredLevel:")
print(df_grouped['physicalTiredLevel'].apply(lambda x: pd.Series(x)).stack().value_counts())

filtered_df.to_csv('filtered_data.csv', index=False)

'''
# Filter IDs with more than 100,000 value counts
filtered_ids = id_counts[id_counts > 100000].index
filtered_df = df[df['id'].isin(filtered_ids)]
print(filtered_df.groupby('id')['id'].value_counts().sort_index())
num_ids = filtered_df['id'].nunique()
print("Number of IDs:", num_ids)
'''
