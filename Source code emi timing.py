import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import recall_score, accuracy_score, mean_absolute_error

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- I. Data Loading (Using the Provided KaggleHub Logic) ---

try:
    # Assuming 'bank_data.csv' is a likely file name based on the dataset name
    file_path = "bank_data.csv"
    
    # NOTE: This call requires the user's environment to be configured for KaggleHub.
    # If the environment is not configured, this will fail, and the mock data will be used.
    print(f"Attempting to load data from KaggleHub at file_path: {file_path}")
    df_raw = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "deepak915/bank-customer-payment-behavior",
        file_path
    )
    print("Data loaded successfully.")

except Exception as e:
    print(f"KaggleHub loading failed or file not found: {e}. Generating synthetic mock data...")
    # --- FALLBACK: Synthetic Data Generation (Mimicking the Expected Structure) ---
    num_loans = 1000
    sequence_length = 12
    loan_ids = np.arange(1, num_loans + 1)
    
    dates = pd.to_datetime(pd.date_range('2024-01-01', periods=sequence_length, freq='M').repeat(num_loans))
    loan_ids_repeated = np.tile(loan_ids, sequence_length)
    
    df_raw = pd.DataFrame({
        'Loan_ID': loan_ids_repeated,
        'Date': dates,
        'DPD_Current': np.random.choice([0, 7, 30, 90], size=len(dates), p=[0.75, 0.15, 0.05, 0.05]), # Days Past Due
        'Payment_Amount': np.random.uniform(500, 1500, size=len(dates)),
        'Scheduled_EMI': np.random.uniform(1000, 1500, size=len(dates)),
        'Days_To_Pay': np.random.randint(1, 31, size=len(dates)), # Actual day of payment (1-30)
    })
    
    # Generate Targets at the loan level
    final_dpd = df_raw.groupby('Loan_ID')['DPD_Current'].max()
    final_pay_day = df_raw.groupby('Loan_ID')['Days_To_Pay'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[-1])

    df_raw['is_defaulter'] = df_raw['Loan_ID'].map((final_dpd >= 30).astype(int))
    df_raw['best_pay_day'] = df_raw['Loan_ID'].map(final_pay_day)

    # Ensure all columns are numeric except ID and Date for the model
    df_raw = df_raw.select_dtypes(include=np.number)

print("First 5 records of processed data:")
print(df_raw.head())

# Define the sequential features
TIME_SERIES_FEATURES = ['DPD_Current', 'Payment_Amount', 'Scheduled_EMI', 'Days_To_Pay']
TARGET_CLASSIFICATION = 'is_defaulter'
TARGET_REGRESSION = 'best_pay_day'
SEQUENCE_LENGTH = df_raw.groupby('Loan_ID').size().mode()[0]
NUM_CUSTOMERS = df_raw['Loan_ID'].nunique()

# --- II. Data Transformation & Scaling for LSTM ---

# 1. Scaling the Time-Series Features
scaler = StandardScaler()
df_scaled = df_raw.copy()
df_scaled[TIME_SERIES_FEATURES] = scaler.fit_transform(df_scaled[TIME_SERIES_FEATURES])

# 2. Reshaping into 3D Array [Samples, Timesteps, Features]
def reshape_to_lstm(df_input, features, sequence_len):
    """Aggregates per-month data into a 3D tensor per Loan_ID."""
    X_list = []
    
    # Group data by Loan_ID
    for _, group in df_input.groupby('Loan_ID'):
        # Ensure the sequence length is consistent (padding/truncating)
        if len(group) >= sequence_len:
            # Take the last 'sequence_len' records
            X_list.append(group[features].values[-sequence_len:])
        elif len(group) > 0:
            # Handle shorter sequences (padding with zeros or the mean)
            padding = np.zeros((sequence_len - len(group), len(features)))
            padded_sequence = np.vstack([padding, group[features].values])
            X_list.append(padded_sequence)
    
    return np.array(X_list)

X_3D = reshape_to_lstm(df_scaled, TIME_SERIES_FEATURES, SEQUENCE_LENGTH)
print(f"\nFinal LSTM Input Shape: {X_3D.shape}")

# 3. Prepare Targets (Need one value per Loan_ID)
df_targets = df_raw.drop_duplicates(subset=['Loan_ID'])
Y_defaulter = df_targets[TARGET_CLASSIFICATION].values
Y_pay_day = df_targets[TARGET_REGRESSION].values

# 4. Scale Regression Target (Day)
day_scaler = MinMaxScaler(feature_range=(0, 1))
Y_pay_day_scaled = day_scaler.fit_transform(Y_pay_day.reshape(-1, 1))

# --- III. Data Splitting ---

# Split the data consistently for both models
X_train, X_test, Y_def_train, Y_def_test, Y_day_train, Y_day_test = train_test_split(
    X_3D, Y_defaulter, Y_pay_day_scaled, test_size=0.2, random_state=42, stratify=Y_defaulter
)

# --- IV. Multi-Task LSTM Model Creation ---

def create_multi_task_lstm(sequence_len, num_feat):
    """Builds a single LSTM model with two output heads."""
    # Input Layer 
    input_layer = Input(shape=(sequence_len, num_feat), name='input_sequence')
    
    # Core LSTM Layer
    lstm_out = LSTM(units=64, activation='relu')(input_layer)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # --- Branch 1: Default Risk (Classification) ---
    def_branch = Dense(32, activation='relu')(lstm_out)
    def_branch = Dense(1, activation='sigmoid', name='defaulter_output')(def_branch)
    
    # --- Branch 2: Payment Day (Regression) ---
    day_branch = Dense(32, activation='relu')(lstm_out)
    day_branch = Dense(1, activation='linear', name='payday_output')(day_branch)
    
    # Create the Model
    model = Model(inputs=input_layer, outputs=[def_branch, day_branch])
    
    # Compile with two losses and metrics
    model.compile(
        optimizer='adam',
        loss={'defaulter_output': 'binary_crossentropy', 'payday_output': 'mse'},
        metrics={'defaulter_output': ['accuracy', tf.keras.metrics.Recall()],
                 'payday_output': ['mae']}
    )
    return model

model = create_multi_task_lstm(SEQUENCE_LENGTH, X_3D.shape[2])

# --- V. Training and Evaluation ---

print("\n--- Training Multi-Task LSTM Model ---")
# Training the model on both targets simultaneously
history = model.fit(
    X_train,
    {'defaulter_output': Y_def_train, 'payday_output': Y_day_train},
    epochs=20, # Use more epochs (e.g., 50-100) for real-world scenarios
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# Predict on the test set
Y_pred_def_proba, Y_pred_day_scaled = model.predict(X_test, verbose=0)

# --- Evaluation 1: Defaulter Prediction ---
Y_pred_def = (Y_pred_def_proba.flatten() > 0.5).astype(int)
def_accuracy = accuracy_score(Y_def_test, Y_pred_def)
def_recall = recall_score(Y_def_test, Y_pred_def)

print("\n## 📈 Default Risk Prediction Metrics")
print(f"**Accuracy:** {def_accuracy:.4f}")
print(f"**Recall (Defaulters identified):** {def_recall:.4f} (Crucial metric for risk management)")

# --- Evaluation 2: Optimal Payment Day Prediction ---
# Inverse transform the scaled prediction back to the 1-30 day range
Y_pred_day_int = day_scaler.inverse_transform(Y_pred_day_scaled).round().clip(1, 30).astype(int)

# Unscale the true day values for MAE calculation
Y_day_test_int = day_scaler.inverse_transform(Y_day_test).round().clip(1, 30).astype(int)

day_mae = mean_absolute_error(Y_day_test_int, Y_pred_day_int)

print("\n## 🗓️ Optimal Payment Day Metrics")
print(f"**Mean Absolute Error (MAE):** {day_mae:.2f} days")
print(f"The predicted optimal day is typically off by {day_mae:.2f} days.")

# --- VI. Final Prediction Example ---

sample_index = 0
sample_input = X_test[[sample_index]]

# Predict using the multi-output model
risk_proba_sample, day_scaled_sample = model.predict(sample_input, verbose=0)

# Process Risk Output
risk_proba = risk_proba_sample[0][0]
customer_risk = 1 if risk_proba > 0.5 else 0
risk_status = "**Defaulter (High Risk)**" if customer_risk == 1 else "**Non-Defaulter (Low Risk)**"

# Process Day Output
predicted_day_scaled = day_scaled_sample[0][0]
predicted_day = int(day_scaler.inverse_transform([[predicted_day_scaled]]).round().clip(1, 30)[0][0])
actual_day = Y_day_test_int[sample_index]

print("\n--- Combined Prediction for Sample Customer ---")
print(f"Customer Risk Status (P={risk_proba:.4f}): {risk_status}")
print(f"Recommended Optimal Payment Day: **Day {predicted_day}** of the month")
print(f"(Actual historical preferred day was Day {actual_day})")