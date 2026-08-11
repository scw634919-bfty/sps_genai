import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_CHECKPOINT = "checkpoints/assignment5_llm"
RL_CHECKPOINT = "checkpoints/assignment5_rl"

EPISODES = 300
LEARNING_RATE = 1e-5


PROMPTS = [
    "What is machine learning?",
    "What is artificial intelligence?",
    "What is reinforcement learning?",
    "What is deep learning?",
    "What is a neural network?",
    "What is natural language processing?",
    "What is a transformer model?",
    "What is generative AI?",
]


# The RL agent chooses one of three response formats.
# Format A is the desired format.
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


def get_device():
    return "cpu"


def main():
    device = get_device()

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_CHECKPOINT
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_CHECKPOINT
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We use A, B, and C as the RL actions.
    # The leading space helps GPT-2 represent them as single tokens.
    action_texts = [" A", " B", " C"]

    action_token_ids = []

    for action_text in action_texts:
        token_ids = tokenizer.encode(
            action_text,
            add_special_tokens=False,
        )

        if len(token_ids) != 1:
            raise ValueError(
                f"{action_text} must map to exactly one token."
            )

        action_token_ids.append(
            token_ids[0]
        )

    action_token_ids = torch.tensor(
        action_token_ids,
        device=device,
    )

    # Freeze most of GPT-2.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # RL post-training updates only the final transformer block
    # and final layer normalization.
    for parameter in model.transformer.h[-1].parameters():
        parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    model.train()

    successes = 0

    for episode in range(
        1,
        EPISODES + 1,
    ):
        question = random.choice(
            PROMPTS
        )

        # The model chooses which response format to use.
        prompt = (
            f"Question: {question}\n"
            f"Choose response format:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        outputs = model(
            **inputs
        )

        logits = outputs.logits[
            0,
            -1,
            action_token_ids,
        ]

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        distribution = (
            torch.distributions.Categorical(
                probs=probabilities
            )
        )

        action_index = distribution.sample()

        selected_format = [
            "A",
            "B",
            "C",
        ][action_index.item()]

        log_probability = (
            distribution.log_prob(
                action_index
            )
        )

        # Reward shaping:
        # Format A is the desired response format.
        if selected_format == "A":
            reward = 10.0
            successes += 1
        else:
            reward = -2.0

        # Policy-gradient loss
        loss = (
            -log_probability
            * reward
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )

        optimizer.step()

        if (
            episode == 1
            or episode % 25 == 0
        ):
            print(
                f"Episode {episode}/{EPISODES} | "
                f"Selected format: {selected_format} | "
                f"Reward: {reward:.1f} | "
                f"Loss: {loss.item():.4f}"
            )

    model.save_pretrained(
        RL_CHECKPOINT
    )

    tokenizer.save_pretrained(
        RL_CHECKPOINT
    )

    print(
        "\nFinished RL post-training"
    )

    print(
        f"Training success rate: "
        f"{successes}/{EPISODES}"
    )

    print(
        f"RL checkpoint saved to "
        f"{RL_CHECKPOINT}"
    )


if __name__ == "__main__":
    main()