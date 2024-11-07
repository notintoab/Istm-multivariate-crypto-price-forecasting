import torch

# data parameters
DATA_PATH = 'data/BTC-USD.csv'
TRAIN_TEST_SPLIT = 0.80
INPUT_SEQUENCE_LENGTH = 100
PREDICTION_LENGTH = 50

# model parameters
HIDDEN_SIZE = 64  
NUM_LAYERS = 2    
DROPOUT = 0.3     
LEARNING_RATE = 0.0005  
INPUT_SIZE = 4 
NUM_EPOCHS = 500  
BATCH_SIZE = 64   

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# output parameters
PREDICTIONS_PATH = 'outputs/predictions.csv'
PLOT_PATH = 'outputs/plot.png'
PLOT_TITLE = 'BTC-USD' 
PLOT_SIZE = (10, 8)  
PLOT_DPI = 300
TIME_LABELS = ['2019', '2020', '2021', '2022', '2023', '2024']