import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('dataset.csv', engine='python')

# map sentiments to 0/1
sentiment_map = {'negative': 0, 'positive': 1}
df['label'] = df['sentiment'].map(sentiment_map)

reviews = df['review'].tolist()
labels = df['label'].tolist()


max_seq_len = 512

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return words[:max_seq_len]

processed_reviews = [preprocess(review) for review in reviews]


all_words = [word for review in processed_reviews for word in review]
vocab = Counter(all_words)
vocab = sorted(vocab, key=vocab.get, reverse=True)[:10000]  
vocab_to_int = {word: i+2 for i, word in enumerate(vocab)} 
vocab_to_int['<PAD>'] = 0
vocab_to_int['<UNK>'] = 1

max_len = max(len(review) for review in processed_reviews)

def encode(review):
    encoded = [vocab_to_int.get(word, 1) for word in review]
    return encoded + [0] * (max_len - len(encoded))  # Pad

encoded_reviews = [encode(review) for review in processed_reviews]

class SentimentDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = torch.tensor(reviews, dtype=torch.long).to(device)  
        self.labels = torch.tensor(labels, dtype=torch.float).to(device) 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

dataset = SentimentDataset(encoded_reviews, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)