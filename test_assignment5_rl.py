import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "checkpoints/assignment5_rl"

FORMAT_LABELS = ["A", "B", "C"]

FORMATS = {
    "A": {
        "start": "That is a great question. ",
        "end": " Let me know if you have any other questions.",
    },
    "B": {
        "start": "Here is the answer. ",
        "end": "",
    },
    "C": {
        "start": "",
        "end": " Hope this helps.",
    },
}


device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT
)

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT
).to(device)

model.eval()


action_texts = [" A", " B", " C"]

action_token_ids = torch.tensor(
    [
        tokenizer.encode(
            action_text,
            add_special_tokens=False,
        )[0]
        for action_text in action_texts
    ],
    device=device,
)


questions = [
    "What is machine learning?",
    "What is reinforcement learning?",
    "What is artificial intelligence?",
    "What is a neural network?",
    "What is generative AI?",
]


successes = 0


for question in questions:
    prompt = (
        f"Question: {question}\n"
        f"Choose response format:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[
        0,
        -1,
        action_token_ids,
    ]

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    best_index = torch.argmax(
        probabilities
    ).item()

    selected_format = FORMAT_LABELS[
        best_index
    ]

    if selected_format == "A":
        successes += 1

    print(
        f"\nQuestion: {question}"
    )

    print(
        f"Format A probability: "
        f"{probabilities[0].item():.4f}"
    )

    print(
        f"Format B probability: "
        f"{probabilities[1].item():.4f}"
    )

    print(
        f"Format C probability: "
        f"{probabilities[2].item():.4f}"
    )

    print(
        f"Selected format: "
        f"{selected_format}"
    )

    example_response = (
        FORMATS[selected_format]["start"]
        + "[generated answer]"
        + FORMATS[selected_format]["end"]
    )

    print(
        f"Example formatted response: "
        f"{example_response}"
    )


print(
    f"\nSuccess rate: "
    f"{successes}/{len(questions)}"
)