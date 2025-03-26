# Third-party libraries
import matplotlib.pyplot as plt

# Function to plot training history
def plot_history(history):
    """
    Plots the training and validation loss over epochs.
    Params:
        history: History object returned by model.fit()
    """
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()