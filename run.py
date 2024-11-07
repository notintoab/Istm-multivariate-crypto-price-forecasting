import torch
import pandas as pd
from data_processing import load_and_preprocess_data, split_sequences
from model import LSTM
from train import training_loop
from outputs import plot_predictions, write_to_csv
from config import (INPUT_SEQUENCE_LENGTH, 
                    INPUT_SIZE,
                    PREDICTION_LENGTH, 
                    HIDDEN_SIZE, 
                    NUM_LAYERS,
                    LEARNING_RATE, 
                    NUM_EPOCHS,
                    DATA_PATH)

#
def main():
    # load and preprocess data
    X_train, X_test, y_train, y_test, ss, mm = load_and_preprocess_data()

    # convert to pytorch tensors and reshape for LSTM input
    X_train_tensors_final = torch.Tensor(X_train).reshape(-1,
                                                         INPUT_SEQUENCE_LENGTH,
                                                         INPUT_SIZE)
    
    X_test_tensors_final = torch.Tensor(X_test).reshape(-1,
                                                        INPUT_SEQUENCE_LENGTH,
                                                        INPUT_SIZE)

    # initialize model and optimizer
    lstm_model = LSTM(num_classes=PREDICTION_LENGTH,
                      input_size=INPUT_SIZE,
                      hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_LAYERS)

    optimiser = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    
    # training loop
    training_loop(NUM_EPOCHS,
                  lstm_model,
                  optimiser,
                  torch.nn.MSELoss(),
                  X_train_tensors_final,
                  torch.Tensor(y_train),
                  X_test_tensors_final,
                  torch.Tensor(y_test))

    df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)  

    df.drop(columns=['Adj Close'], inplace=True)

    df_X_ss_full = ss.transform(df.drop(columns=['Close'])) 
    df_y_mm_full = mm.transform(df.Close.values.reshape(-1, 1)) 

    df_X_ss_full, df_y_mm_full = split_sequences(df_X_ss_full, df_y_mm_full, 100, 50)

    df_X_ss_full_tensor = torch.Tensor(df_X_ss_full).reshape(-1, 100, INPUT_SIZE)

    train_predict = lstm_model(df_X_ss_full_tensor).data.numpy() 
    dataY_plot = df_y_mm_full  

    data_predict = mm.inverse_transform(train_predict) 
    dataY_plot = mm.inverse_transform(dataY_plot)

    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])
    
    #
    plot_predictions(true, preds)
    
    #
    write_to_csv(true, preds)

if __name__ == "__main__":
     main()