# Local imports
from utils.preprocessing import preprocess_text, nlp

# Third party imports
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('model/fcnn.keras')

# Load the saved vectorizer
vectorizer = joblib.load('model/vectorizer.joblib')

# Load the saved label encoder
label_encoder = joblib.load('model/label_encoder.joblib')

# Create a new phrase to test the model
phrase = "Sometimes i get lost in my thoughts and i feel so scared"

# Create a new dataframe with the phrase
df_new = pd.DataFrame({'text': [phrase]})

# Apply preprocessing
df_new['text'] = df_new['text'].apply(lambda x: preprocess_text(x, nlp))

# Proccess the new phrase with tfidf
X_new = vectorizer.transform(df_new['text'])

# Convert to array
X_new = X_new.toarray()

# Make predictions
prediction = loaded_model.predict(X_new)

# Extract the predicted classes
bigger_prob = np.argmax(prediction, axis=1)

# Convert the predicted classes to labels
class_names = label_encoder.inverse_transform(bigger_prob)

# Print the predicted class
print(f"The predicted class is: {class_names[0]}")