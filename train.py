import torch
import torch.nn as nn
import torch.optim as optim

from data import vocab_to_int, dataloader
from model import TransformerSentiment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vocab_size = len(vocab_to_int)
model = TransformerSentiment(vocab_size, max_len=512)
model.to(device) 

model_save_path = 'model.pth'
optimizer_save_path = 'optimizer.pth'


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for reviews, labels in dataloader:

        optimizer.zero_grad()
        src_mask = (reviews == 0)
        outputs = model(reviews, src_mask).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    print(f"Checkpoint saved at step {iter}.")
