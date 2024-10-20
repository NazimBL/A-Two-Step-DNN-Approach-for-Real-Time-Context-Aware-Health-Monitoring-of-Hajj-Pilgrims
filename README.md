# A-Two-Step-DNN-Approach-for-Real-Time-Context-Aware-Health-Monitoring-of-Hajj-Pilgrims
 A two-step LSTM-TabNet model that first captures temporal dependencies with an LSTM and then uses TabNet for robust feature selection and classification

Code workflow:
-cleaned_csv comes from the public dataset available in kaggle at the following link:
https://www.kaggle.com/datasets/sohagalalaldinahmed/hajj-crowd-activity-prediction-dataset/data?select=cleaned_data.csv

-Download cleaned_csv by selecting the columns of interest:
'gsr_x', 'altitude', 'peakAcceleration', 'ibi', 'temp', 'x', 'y', 'z','heartRate', 'respirationRate', 'heartRateVariability','physicalTiredLevel', 'emotionalMoodLevel'

-Run the Preprocessing.py:
Iniitially, we handle anomalies by identifying and addressing outliers and missing values. Using a Z-score normalization method, we standardize the physiological measurements to ensure consistency. Anomalies in the physiological data (e.g.,temperature, heart rate, and respiration rate) were identified and corrected, with outliers replaced by the mean of the 10 nearest non-anomalous values to maintain data integrity and minimize noise. The data is then grouped by participant IDs, selecting the ID with the least amount of anomalies for further analysis.
Features are aggregated into sequences for each participant, using the numerical features: GSR, altitude, peak acceleration, IBI, TEMP, X, Y, Z, HR, respiration rate, and HRV. This preprocessing step ensures that the dataset is clean and standardized, ready for use in the subsequent stages of our proposed method.

-Physical Tiredness Level: Run TwoStep_physicalTiredLevel_Prediction.py
The Physical Tiredness Level classification task involved predicting one of five possible values, ranging from 1 (not tired at all) to 5 (extremely tired).

-Emotional Mood Level: Run TwoStep_emotionalMoodLevel_Prediction.py
The Emotional Mood Level classification task aimed to predict the emotional status of the participants, categorized into three classes: 0 (very negative and negative), 1 (neutral), and 2 (positive and very positive).

-Rukun (Hajj Ritual) Activity: Run TwoStep_rukun_Prediction.py
The Rukun (Hajj Ritual) Activity classifica￾tion task involved predicting one of 16 specific Hajj-related activities performed by the participants.


Feel free to reach out to me at bellabaci.nazim@gmail.com if you have any question or feedback.

If you like this work or if it this code was usefull to you in anyway, please cite our paper =)
