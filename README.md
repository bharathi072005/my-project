# my-project
import pandas as pd

# Sample social media posts and emotions
data = {
    'text': [
        "I just got a promotion at work!",
        "I'm feeling really low today.",
        "That jump scare was terrifying!",
        "What a beautiful wedding.",
        "I'm furious about the delay in my flight.",
        "Just won the lottery!",
        "Missed my bus again. Ugh!",
        "Why did this happen to me?",
        "OMG I can't believe it!",
        "The sunset is breathtaking.",
        "I'm so anxious for the interview.",
        "This made my day!",
        "Everything is ruined.",
        "Totally unexpected surprise party!",
        "I hate when people lie."
    ],
    'emotion': [
        'joy', 'sadness', 'fear', 'joy', 'anger',
        'joy', 'anger', 'sadness', 'surprise', 'joy',
        'fear', 'joy', 'sadness', 'surprise', 'anger'
    ]
}

# Create and save DataFrame
df = pd.DataFrame(data)
df.to_csv('social_media_emotions.csv', index=False)

print("Sample dataset created as 'social_media_emotions.csv'")

df = pd.read_csv('social_media_emotions.csv')
print(df.head())

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv("social_media_emotions.csv")
print("Original shape:", df.shape)

# Drop missing and duplicate entries
df.dropna(subset=['text', 'emotion'], inplace=True)
df.drop_duplicates(subset='text', inplace=True)

# Filter out very short texts
df = df[df['text'].apply(lambda x: len(str(x).split()) > 2)]

# Define cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Final output
print("Cleaned shape:", df.shape)
print(df[['cleaned_text', 'emotion']].head())  # show emotion instead of encoded_emotion
