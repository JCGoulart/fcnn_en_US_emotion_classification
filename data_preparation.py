# Imports
import pandas as pd

# Internal imports
from utils.preprocessing import preprocess_text, nlp

# This script prepares the data for the fully connected neural network
# The dataset used is the Emotion Dataset from Hugging Face
# The dataset is used for emotion classification

# The dataset is available at https://huggingface.co/datasets/dair-ai/emotion
# Set train, validation and test splits
splits = {'train': 'split/train-00000-of-00001.parquet', 
          'validation': 'split/validation-00000-of-00001.parquet', 
          'test': 'split/test-00000-of-00001.parquet'}

# Load the datasets
train_data = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
test_data = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])
validation_data = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["validation"])

# Rename the labels in the dataframes
labels = {0: 'sadness',
          1: 'joy',
          2: 'love',
          3: 'anger',
          4: 'fear',
          5: 'surprise'}

# Map the labels to the corresponding values
train_data['label'] = train_data['label'].map(labels)
test_data['label'] = test_data['label'].map(labels)
validation_data['label'] = validation_data['label'].map(labels)

# Preprocess the text data
train_data['text'] = train_data['text'].apply(lambda x: preprocess_text(x, nlp))
test_data['text'] = test_data['text'].apply(lambda x: preprocess_text(x, nlp))
validation_data['text'] = validation_data['text'].apply(lambda x: preprocess_text(x, nlp))

# Save the dataframes to parquet files
train_data.to_parquet("data/train_data.parquet")
test_data.to_parquet("data/test_data.parquet")
validation_data.to_parquet("data/validation_data.parquet")