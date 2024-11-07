import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import DATA_PATH, INPUT_SEQUENCE_LENGTH, PREDICTION_LENGTH, TRAIN_TEST_SPLIT

#
def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    df.drop(columns=['Adj Close'], inplace=True)  

    X, y = df.drop(columns=['Close']), df.Close.values

    mm = MinMaxScaler()
    ss = StandardScaler()

    X_trans = ss.fit_transform(X)  
    y_trans = mm.fit_transform(y.reshape(-1, 1))

    X_ss, y_mm = split_sequences(X_trans, y_trans, INPUT_SEQUENCE_LENGTH, PREDICTION_LENGTH)

    total_samples = len(X)

    train_test_cutoff = round(TRAIN_TEST_SPLIT * total_samples)

    X_train = X_ss[:-150]
    X_test = X_ss[-150:]

    y_train = y_mm[:-150]
    y_test = y_mm[-150:]

    return X_train, X_test, y_train, y_test, ss, mm

# 
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(input_sequences): break
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)