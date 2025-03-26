# Third-party imports
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to evaluate the model on the test set and save it
def evaluate_and_save_model(model, X_test, y_test_encoded):
    """
    Evaluate the model on the test set and save it.
    Params:
        model: The trained model to evaluate.
        X_test: The test features.
        y_test_encoded: The encoded test labels.
    """

    # Make predictions on the test set
    prediction = model.predict(X_test)
    # Extract the predicted classes
    labels = prediction.argmax(axis=1)
    
    # Generate and display reports
    print("Classification Report:")
    print(classification_report(y_test_encoded, labels))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_encoded, labels))
    
    print("Accuracy Score:", accuracy_score(y_test_encoded, labels))
    
    # Save the model
    model.save('model/fcnn.keras')