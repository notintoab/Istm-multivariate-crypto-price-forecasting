import matplotlib.pyplot as plt
from config import PREDICTIONS_PATH, PLOT_PATH, PLOT_TITLE, PLOT_SIZE, PLOT_DPI

# 
def plot_predictions(true_values, predicted_values):
    plt.figure(figsize=PLOT_SIZE)
    
    plt.axvline(x=len(true_values) - 150, c='r', linestyle='--')
    
    plt.plot(true_values, label='Actual Data') 
    plt.plot(predicted_values, label='Predicted Data') 
    plt.title(PLOT_TITLE)
    
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    plt.legend()
    
    plt.savefig(PLOT_PATH, dpi=PLOT_DPI) 
    plt.show()

# 
def write_to_csv(true_values, predicted_values):
    true_values = true_values[-150:]
    predicted_values = predicted_values[-150:]
    
    with open(PREDICTIONS_PATH, 'w') as f:
        f.write('Actual_Price, Predicted_Price\n')
        for true_val, pred_val in zip(true_values, predicted_values):
            f.write(f'{true_val},{pred_val}\n')