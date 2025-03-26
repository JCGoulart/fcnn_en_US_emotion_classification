# Standard imports
import math

# Local imports
from utils.evaluate_and_save import evaluate_and_save_model
from utils.training_plot import plot_history

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Define the step decay function before using it in the callback
def step_decay(epoch):
    """
    Step decay function to adjust the learning rate.
    This function reduces the learning rate by a factor of `drop` every `epochs_drop` epochs.
    The initial learning rate is set to `initial_lrate`.
    
    Parameters:
        epoch (int): Current epoch number.
    
    Returns:
        float: Adjusted learning rate.
    """
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

# Callback to schedule the learning rate with step decay
lrate_scheduler = LearningRateScheduler(step_decay)

# Callback to EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=3,
)

# Function to build the model
def build_model(input_dim, class_weights):
    model = Sequential()
    model.add(
        Dense(
            1024,
            activation='selu',
            kernel_initializer='lecun_normal',
            input_shape=(input_dim, ),
            kernel_regularizer=l2(0.01)
        )
    )
    model.add(Dense(512, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(256, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='selu'))
    model.add(Dense(6, activation='softmax'))
    # Adjust output bias weights using the computed class weights
    model.layers[-1].bias.assign(class_weights.astype(np.float32))
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load data
    train_data = pd.read_parquet('data/train_data.parquet')
    test_data = pd.read_parquet('data/test_data.parquet')
    validation_data = pd.read_parquet('data/validation_data.parquet')

    # Create feature matrices for training, validation and test sets
    # Vectorizer with TF-IDF
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    X_val = vectorizer.transform(validation_data['text'])

    # Export the vectorizer for later use
    joblib.dump(vectorizer, 'model/vectorizer.joblib')

    # Convert the sparse matrices to dense matrices
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    X_val = X_val.toarray()

    # Codification of the labels in numeric values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_data['label'])
    y_test_encoded = label_encoder.transform(test_data['label'])
    y_val_encoded = label_encoder.transform(validation_data['label'])

    # Export the label encoder for later use
    joblib.dump(label_encoder, 'model/label_encoder.joblib')

    # Calculate weights for the classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_encoded),
        y=y_train_encoded
    )

    # Convert the labels to categorical (one-hot encoding)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)
    y_val_categorical = to_categorical(y_val_encoded)

    # Build the model using the input dimension from the training features
    model = build_model(X_train.shape[1], class_weights)

    # Training hyperparameters
    num_epochs = 20
    batch_size = 256

    # Train the model
    history = model.fit(
        X_train,
        y_train_categorical,
        validation_data=(X_val, y_val_categorical),
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lrate_scheduler]
    )

    # Plot the training history
    plot_history(history)

    # Evaluate the model and save it
    evaluate_and_save_model(model, X_test, y_test_encoded)