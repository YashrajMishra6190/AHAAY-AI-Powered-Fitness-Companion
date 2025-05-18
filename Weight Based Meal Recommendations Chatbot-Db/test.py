from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the trained model and tokenizer
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

# Generate a response from the model
def generate_response(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=32, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model_path = "./saved_model"  # Path to the saved model
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Example user input
    user_input = "Question: What should I eat in the morning?\nCategory: Male_Overweight"

    # Generate response
    response = generate_response(model, tokenizer, user_input)
    print("Model Response:", response)

if __name__ == "__main__":
    main()
