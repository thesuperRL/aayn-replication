from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenized_datasets():
    dataset_iwslt = load_dataset("bentrevett/multi30k")

    train_data = dataset_iwslt["train"]
    val_data = dataset_iwslt["validation"]
    test_data = dataset_iwslt["test"]

    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    de_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-german-cased")

    src_vocab_size = en_tokenizer.vocab_size
    tgt_vocab_size = de_tokenizer.vocab_size
    pad_token_id = de_tokenizer.pad_token_id

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

    tokenized_train = train_data.map(
        preprocess_function, batched=True, remove_columns=["en", "de"]
    )
    tokenized_val = val_data.map(
        preprocess_function, batched=True, remove_columns=["en", "de"]
    )
    tokenized_test = test_data.map(
        preprocess_function, batched=True, remove_columns=["en", "de"]
    )

    tokenized_train.set_format(type="torch")
    tokenized_val.set_format(type="torch")
    tokenized_test.set_format(type="torch")
    
    return tokenized_train, tokenized_val, tokenized_test, src_vocab_size, tgt_vocab_size, pad_token_id

SRC_VOCAB_SIZE = 30522
TGT_VOCAB_SIZE = 28996
PAD_TOKEN_ID = 0