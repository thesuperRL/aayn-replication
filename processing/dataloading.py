from torch.utils.data import DataLoader
from .tokenization import get_tokenized_datasets

def get_dataloaders(batch_size=32):
    tokenized_train, tokenized_val, tokenized_test, src_vocab, tgt_vocab, pad_id = get_tokenized_datasets()
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab, pad_id