We Simply Import The Necessary Libraries. In This Case We Install 'Transformers' and also 'Emoji' For emoji Conversion.

!pip install -q transformers  # Install the transformers library

!pip install emoji==0.6.0  # Install emoji library version 0.6.0

RUN SENTIMENT ANALYSIS PREDICTIONS USING PIPELINE

from transformers import pipeline  # Import the pipeline module from transformers library
import emoji  # Import the emoji library for emoji conversion

# Handling the model selection explicitly 
# Define mapping dictionary for the first pipeline
label_mapping = {
    "LABEL_1": "positive",
    "LABEL_2": "negative",
    "LABEL_3": "neutral",
    # Add more labels as needed
}

# Function to convert labels for the first pipeline
def convert_labels(result):
    """
    Convert the sentiment label from LABEL_1, LABEL_2, etc. to positive, negative, neutral.
    
    Args:
        result (dict): Dictionary containing the sentiment analysis result.

    Returns:
        dict: Dictionary with the label converted.
    """
    label = result["label"]
    if label in label_mapping:
        result["label"] = label_mapping[label]
    return result

# Create sentiment analysis pipelines
sentiment_pipeline = pipeline("sentiment-analysis")  # Create a sentiment analysis pipeline using a default model
specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")  # Create a sentiment analysis pipeline using a specific model

data = [  # Input data for sentiment analysis
    "I love you😍 i will never love you less",
    "I hate you,there is really no doubt about it😒😒😒",
    "The news is really exciting👌👌 The sportsman were playing really well!",
    "I don't know if i am happy or sad😶"
]
sentiment_pipeline(data)  # Perform sentiment analysis using the sentiment_pipeline

Use a specific sentiment analysis model available on Hub by specifying its name

# Performing sentiment analysis with the sentiment_pipeline
print("Sentiment analysis with distilbert-base-uncased model:")  # Print a header indicating the model being used
for i, result in enumerate(sentiment_pipeline(data)):  # Iterate over the sentiment analysis results from sentiment_pipeline
    print(f"Sentence: {data[i]}")  # Print the input sentence for the current result
    print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})")  # Print the sentiment label and score for the current result
    print()  # Print a newline character to separate the output for each input sentence

# Performing sentiment analysis with the specific_model
print("Sentiment analysis with finiteautomata/bertweet-base-sentiment-analysis model:")  # Print a header indicating the model being used
for i, result in enumerate(specific_model(data)):  # Iterate over the sentiment analysis results from specific_model
    print(f"Sentence: {data[i]}")  # Print the input sentence for the current result
    print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})")  # Print the sentiment label and score for the current result
    print()  # Print a newline character to separate the output for each input sentence
