import pymongo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

# Connect to MongoDB and retrieve data
def get_data_from_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["chatbot-db"]
    data = []

    collections = [
        "Male_Underweight", "Male_Normalweight", "Male_Overweight", "Male_Obesity",
        "Female_Underweight", "Female_Normalweight", "Female_Overweight", "Female_Obesity"
    ]

    for collection_name in collections:
        collection = db[collection_name]
        for record in collection.find():
            data.append({
                "input_text": f"Question: {record['question']}",
                "output_text": f"Answer: {record['answer']}",
                "category": collection_name
            })

    return data

# Prepare the dataset
def prepare_dataset(data):
    dataset = Dataset.from_list(data)
    return dataset

# Load model and tokenizer for a text-only model
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# Tokenize the dataset
def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(examples["input_text"], max_length=32, truncation=True, padding="max_length")
    labels = tokenizer(examples["output_text"], max_length=32, truncation=True, padding="max_length")

    # Shift the labels to the right for teacher forcing
    labels["input_ids"] = [[(label if label != tokenizer.pad_token_id else -100) for label in lbl] for lbl in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Main training function
def main():
    model_name = "t5-base"  # Use a text-only model
    data = get_data_from_mongodb()
    dataset = prepare_dataset(data)
    model, tokenizer = load_model_and_tokenizer(model_name)

    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # Reduce batch size if needed
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

if __name__ == "__main__":
    main()
