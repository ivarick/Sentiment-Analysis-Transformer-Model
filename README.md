# Sentiment Analysis Transformer Model

This repository contains PyTorch-based scripts for training and predicting with a Transformer model on sentiment analysis tasks, using a dataset of reviews labeled as positive or negative. The model processes text data, builds a vocabulary, encodes reviews, and trains a custom Transformer architecture for binary classification.

## Features
- Data preprocessing: Lowercasing, punctuation removal, truncation to a maximum sequence length.
- Vocabulary building with optional size limit (e.g., top 10,000 words).
- Positional encoding for Transformer inputs.
- Training loop with BCE loss and Adam optimizer.
- Inference function for predicting sentiment on new text.

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Regular expressions (built-in)

Install dependencies via:
```
pip install torch numpy pandas
```

## Dataset
The script expects a CSV file named `dataset.csv` with columns:
- `review`: The text of the review.
- `sentiment`: Label as 'positive' or 'negative'.

Example structure:
```
review,sentiment
"This movie was great!",positive
"I hated this film.",negative
```

## Usage
1. Place your `dataset.csv` in the same directory as the scripts.

2. Train the model (trains for 50 epochs):
   ```
   python train.py
   ```

3. Use the prediction script for inference:
   ```
   python predict.py
   ```
```python
print(predict_sentiment("best movie ever"))  # Expected: Positive
print(predict_sentiment("besides being boring, the scenes were oppressive and dark."))   # Expected: Negative
print(predict_sentiment("not a really good acting"))   # Expected: Negative
print(predict_sentiment("2/10"))   # Expected: Negative
print(predict_sentiment("9/10"))  # Expected: Positive
```


## Model Definition
The `TransformerSentiment` class is a custom Transformer implementation using `nn.TransformerEncoder`. It includes embedding, positional encoding, encoder layers, and a final linear layer with sigmoid activation.

Example snippet:
```python
class TransformerSentiment(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, src, src_mask):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0, 1), src_key_padding_mask=src_mask)
        output = output.mean(dim=0)  # Global average pooling
        output = self.fc(output)
        return self.sigmoid(output)
```

## Limitations
- The model assumes binary sentiment; extend for multi-class if needed.
- Performance depends on the dataset quality and size.
- No validation split; add for better evaluation.

## Contributing
Feel free to fork and submit pull requests for improvements, such as hyperparameter tuning.

## License
MIT License
