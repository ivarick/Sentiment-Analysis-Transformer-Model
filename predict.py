import torch
import torch.nn as nn
from data import vocab_to_int, encode, preprocess
from model import TransformerSentiment


vocab_size = len(vocab_to_int)
model = TransformerSentiment(vocab_size, max_len=512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)


def predict_sentiment(text):
    model.eval()
    processed = preprocess(text)
    encoded = encode(processed)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)  # Move to device
    src_mask = (input_tensor == 0)
    with torch.no_grad():
        output = model(input_tensor, src_mask).squeeze().item()
    return "Positive" if output > 0.5 else "Negative"


print(predict_sentiment("best movie ever"))  # Expected: Positive
print(predict_sentiment("besides being boring, the scenes were oppressive and dark."))   # Expected: Negative
print(predict_sentiment("not a really good acting"))   # Expected: Negative
print(predict_sentiment("2/10"))   # Expected: Negative
print(predict_sentiment("9/10"))  # Expected: Positive
