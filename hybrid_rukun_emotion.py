import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from keras.utils import to_categorical

df = pd.read_csv('filtered_data.csv')

label_encoder = LabelEncoder()
# Fit and transform the 'rukun' column
df['rukun_encoded'] = label_encoder.fit_transform(df['rukun'])

# Define the mapping for the emotional mood levels
mapping = {
    'very negative': 0,
    'negative': 0,
    'neutral': 0,
    'positive': 1,
    'very positive': 2
}
# Map the emotional mood levels to the desired class labels
df['emotionalMoodLevel'] = df['emotionalMoodLevel'].map(mapping)


# Group by ID and calculate the mean physicalTiredLevel for each ID
df_grouped = df.groupby('id').agg({
    'gsr_x': list, 'altitude': list, 'peakAcceleration': list,
    'ibi': list, 'temp': list, 'x': list, 'y': list, 'z': list,
    'heartRate': list, 'respirationRate': list, 'heartRateVariability': list,'rukun_encoded':list,
    'emotionalMoodLevel': list
}).reset_index()

print(df_grouped.shape)
print(df['emotionalMoodLevel'].value_counts())
# Initialize lists to store X_lstm and y_lstm
X_lstm = []
y_lstm = []


numerical_features = ['gsr_x', 'altitude', 'peakAcceleration', 'ibi', 'temp', 'x', 'y', 'z', 'heartRate', 'respirationRate', 'heartRateVariability','rukun_encoded' ]

sequence_length = 50
num_features = len(numerical_features)

for idx, row in df_grouped.iterrows():
    id_data = [np.array(row[feature]) for feature in numerical_features]
    id_matrix = np.column_stack(id_data)
    if id_matrix.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - id_matrix.shape[0], num_features))
        id_matrix_padded = np.vstack([id_matrix, padding])
        X_lstm.append(id_matrix_padded)
        y_lstm_flat = [row['emotionalMoodLevel'][0]] * (id_matrix.shape[0] // sequence_length)
        y_lstm.extend(y_lstm_flat)
    else:
        for start in range(0, id_matrix.shape[0] - sequence_length + 1, sequence_length):
            end = start + sequence_length
            X_lstm.append(id_matrix[start:end])
            y_lstm.append(row['emotionalMoodLevel'][start])

y_lstm = np.array(y_lstm)

# Convert emotionalMoodLevel to categorical
y_lstm_categorical = to_categorical(y_lstm, num_classes=3)
# Print the number of instances in each class
for class_label in np.unique(y_lstm):
    class_count = np.sum(y_lstm== class_label)
    print(f"Class {class_label}: {class_count} instances")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))


# Reshape for scaling and then reshape back
X_lstm = np.array(X_lstm)
X_lstm_reshaped_for_scaling = X_lstm.reshape(-1, num_features)
X_lstm_normalized = scaler.fit_transform(X_lstm_reshaped_for_scaling)
X_lstm_normalized = X_lstm_normalized.reshape(X_lstm.shape)


# Split the data

# Split the data
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm  = train_test_split(
    X_lstm_normalized, y_lstm_categorical, test_size=0.33, random_state=42, stratify=y_lstm_categorical
)
# Define and train the LSTM model with an additional LSTM layer
model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(sequence_length, num_features)))
model.add(LSTM(16))  # Add another LSTM layer
model.add(Dense(3, activation='softmax'))  # Use 5 units for the 5 classes and softmax activation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with EarlyStopping
history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=10,  # Set the maximum number of epochs
    batch_size=32,
    validation_data=(X_val_lstm, y_val_lstm),
    callbacks=[early_stopping]  # Add the EarlyStopping callback
)

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('train_vs_validation.png')

# Extract LSTM embeddings
lstm_embeddings = model.predict(X_lstm_normalized)

y_pred = model.predict(X_val_lstm)


# Reshape lstm_embeddings and y_lstm for TabNet
lstm_embeddings_reshaped = lstm_embeddings.reshape(lstm_embeddings.shape[0], -1)
y_lstm_reshaped = y_lstm.flatten()


print(lstm_embeddings_reshaped.shape)
print(y_lstm_reshaped.shape)

# Split the normalized data for TabNet
X_train_tabnet, X_val_tabnet, y_train_tabnet, y_val_tabnet = train_test_split(
    lstm_embeddings_reshaped, y_lstm_reshaped, test_size=0.2, random_state=42
)

tabnet_params = dict(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.3,
    n_independent=2,
    n_shared=2,
    epsilon=1e-15,
    momentum=0.02,
    scheduler_params=dict(
        max_lr=0.01,
        steps_per_epoch=len(X_train_tabnet),
        epochs=30,
        is_batch_level=True
    ),
    verbose=1
)

tabnet_model = TabNetClassifier(**tabnet_params)

# Train the TabNet model
print(X_val_tabnet.shape)
print(y_val_tabnet.shape)
tabnet_history = tabnet_model.fit(X_train_tabnet, y_train_tabnet, eval_set=[(X_val_tabnet, y_val_tabnet)], patience=10, max_epochs=30)

# Predict using the trained TabNet model
y_pred_tabnet = tabnet_model.predict(X_val_tabnet)

#Evaluate classifier
accuracy = accuracy_score(y_val_tabnet, y_pred_tabnet)
print(f"Validation Accuracy: {accuracy}")

f1 = f1_score(y_val_tabnet, y_pred_tabnet, average='weighted')
print(f"F1-score: {f1}")

precision = precision_score(y_val_tabnet, y_pred_tabnet, average='weighted')
print(f"Precision: {precision}")

recall = recall_score(y_val_tabnet, y_pred_tabnet, average='weighted')
print(f"Recall: {recall}")

classification_rep = classification_report(y_val_tabnet, y_pred_tabnet)
print(f"Classification Report:\n{classification_rep}")

'''''
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm  = train_test_split(
    X_lstm_normalized, y_lstm_categorical, test_size=0.2, random_state=42
)
'''