import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "openai-community/gpt2"
CHECKPOINT_DIR = "checkpoints/assignment5_llm"


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(device)

    dataset = load_dataset(
        "rajpurkar/squad",
        split="train[:100]"
    )

    def tokenize_example(example):
        answer = (
            example["answers"]["text"][0]
            if example["answers"]["text"]
            else ""
        )

        text = (
            f"Question: {example['question']}\n"
            f"Answer: {answer}"
        )

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    tokenized_dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=2,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
    )

    model.train()

    for batch_index, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        print(
            f"Batch {batch_index + 1}/{len(train_loader)} "
            f"| Loss: {loss.item():.4f}"
        )

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    print(
        f"LLM checkpoint saved to {CHECKPOINT_DIR}"
    )


if __name__ == "__main__":
    main()