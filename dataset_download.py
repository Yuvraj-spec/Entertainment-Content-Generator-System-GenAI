from datasets import load_dataset
import json

# Load the dataset from Hugging Face
dataset = load_dataset("rohitsaxena/MovieSum")

print("Dataset loaded successfully.")
print(dataset)

# Function to save dataset split safely (row by row)
def save_split(split_name, filename):
    print(f"Saving {split_name} split...")

    with open(filename, "w", encoding="utf-8") as f:
        for row in dataset[split_name]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{split_name} saved to {filename}")


# Save each split
save_split("train", "moviesum_train.jsonl")
save_split("validation", "moviesum_validation.jsonl")
save_split("test", "moviesum_test.jsonl")

print("All dataset splits saved successfully.")