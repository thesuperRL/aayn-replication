from datasets import load_dataset
from transformers import AutoTokenizer

# Alternative dataset because the original is too large
dataset_iwslt = load_dataset("bentrevett/multi30k")

train_data = dataset_iwslt["train"]
val_data = dataset_iwslt["validation"]
test_data = dataset_iwslt["test"]

# Borrow tokenizers, since that is not the focus of this excercise
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
de_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-german-cased")

def preprocess_function(examples):
    en_texts = examples["en"]
    de_texts = examples["de"]
    
    src_tokenized = en_tokenizer(en_texts, padding="max_length", truncation=True, max_length=128)
    tgt_tokenized = de_tokenizer(de_texts, padding="max_length", truncation=True, max_length=128)
    
    return {
        "input_ids": src_tokenized["input_ids"],
        "attention_mask": src_tokenized["attention_mask"],
        "labels": tgt_tokenized["input_ids"]
    }

# Map the tokenizers over the splits in parallel batches
tokenized_train = train_data.map(
    preprocess_function, batched=True, remove_columns=["en", "de"]
)
tokenized_test = test_data.map(
    preprocess_function, batched=True, remove_columns=["en", "de"]
)

# Automatically cast everything to PyTorch Tensors
tokenized_train.set_format(type="torch")
tokenized_test.set_format(type="torch")