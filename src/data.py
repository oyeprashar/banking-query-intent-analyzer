"""
Things to do for this file :
    1. Create train/validation split
    2. Load the DistilBERT tokenizer
    3. Tokenize queries
    4. Prepare model ready data {input_ids, attention_mask, label}
    5. Expose prepared splits
"""

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer


def load_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

def load_banking77():
    banking77 = load_dataset("banking77")
    train_dataset = banking77["train"]
    test_dataset = banking77["test"]
    return train_dataset, test_dataset

def split_train_validation(train_dataset, validation_size = 0.1, random_state = 42):

    """
        Before tokenization:

            {
                text:  "Why hasn't my card arrived?",
                label: 11
            }

        ------------------------------------------------------
                ↓↓↓↓↓↓ tokenizer ↓↓↓↓↓↓
        ------------------------------------------------------

        After tokenization:

            {
                input_ids:      [101, 2339,  ...],
                attention_mask: [1, 1, 1, ...],
                label:           11
            }

    We club the text and label together because that's what PyTorch/Hugging Face training pipeline wants

    The original training/test data already contains the data clubbed together
    """

    train_indices, validation_indices = train_test_split(
        range(len(train_dataset)), # gives a range object which signifies an array from [0, len(train_dataset) - 1]
        test_size=validation_size,
        random_state=random_state,
        stratify=train_dataset["label"]
    )

    train_split = train_dataset.select(train_indices)
    validation_split = train_dataset.select(validation_indices)

    return train_split, validation_split

def tokenize_batch(batch, tokenizer):

    # DistilBERT can produce 512 tokens. In case the query string is producing more than that we cut off instead of crashing
    return tokenizer(batch["text"], truncation = True)

def tokenize_dataset(dataset, tokenizer):

    """
    The dataset that we are working with is hugging face and it contains tens of thousands of rows
    We can work naively using for loop but that is going to take more time

    The data-structure that hugging face uses is called arrow and they have a technique called "maps" to efficiently
    batch process these tens of thousands of rows
    """

    tokenized_dataset = dataset.map(
        tokenize_batch, # this needs to be a method, we will define it
        batched = True,
        fn_kwargs={"tokenizer": tokenizer}
    )


    return tokenized_dataset

def prepare_data():

    train_dataset, test_dataset = load_banking77()
    train_dataset, validation_dataset = split_train_validation(train_dataset)
    tokenizer = load_tokenizer()


    # Convert the data into tokens that a transformer will understand
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    validation_dataset = tokenize_dataset(validation_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)

    return train_dataset, validation_dataset, test_dataset, tokenizer






