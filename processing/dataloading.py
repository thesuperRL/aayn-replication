from torch.utils.data import DataLoader
from tokenization import tokenized_train, tokenized_test

# Create the iterable batch loaders for training and testing
train_loader = DataLoader(tokenized_train, batch_size=32, shuffle=True)
test_loader = DataLoader(tokenized_test, batch_size=32, shuffle=False)

# Let's inspect a single batch to see what your model will receive
for batch in train_loader:
    print("Encoder Input Shape (Batch Size, Sequence Length):", batch["input_ids"].shape)
    print("Decoder Target Shape (Batch Size, Sequence Length):", batch["labels"].shape)
    
    # Print a tiny sample of what the word IDs look like
    print("\nFirst sentence token IDs in Encoder:\n", batch["input_ids"][0][:10])
    break