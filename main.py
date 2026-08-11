import io

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from helper_lib.model import get_model
from helper_lib.generator import generate_energy_samples
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Assignment 5 RL model
assignment5_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

assignment5_checkpoint = "checkpoints/assignment5_rl"

assignment5_tokenizer = AutoTokenizer.from_pretrained(
    assignment5_checkpoint
)

assignment5_model = AutoModelForCausalLM.from_pretrained(
    assignment5_checkpoint
).to(assignment5_device)

assignment5_model.eval()

ASSIGNMENT5_FORMAT_LABELS = ["A", "B", "C"]

ASSIGNMENT5_FORMATS = {
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

assignment5_action_texts = [" A", " B", " C"]

assignment5_action_token_ids = torch.tensor(
    [
        assignment5_tokenizer.encode(
            action_text,
            add_special_tokens=False,
        )[0]
        for action_text in assignment5_action_texts
    ],
    device=assignment5_device,
)

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
try:
    embedding_model = EmbeddingModel()
except OSError:
    embedding_model = None

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

assignment4_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

cnn_device = torch.device("cpu")
cnn_model = get_model("AssignmentCNN")

checkpoint = torch.load(
    "checkpoints/assignment_cnn/best/model_epoch_001.pth",
    map_location=cnn_device,
)

cnn_model.load_state_dict(checkpoint["model_state_dict"])
cnn_model.to(cnn_device)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Assignment 3 MNIST GAN setup
mnist_gan_device = torch.device("cpu")
mnist_gan_model = get_model("MNISTGAN")

mnist_gan_checkpoint = torch.load(
    "checkpoints/assignment3_gan/mnist_gan.pth",
    map_location=mnist_gan_device
)

mnist_gan_model.load_state_dict(mnist_gan_checkpoint["model_state_dict"])
mnist_gan_model.to(mnist_gan_device)
mnist_gan_model.eval()

diffusion_model = get_model("Diffusion").to(assignment4_device)

diffusion_checkpoint = torch.load(
    "checkpoints/diffusion/diffusion_cifar10.pth",
    map_location=assignment4_device,
)

diffusion_model.load_state_dict(
    diffusion_checkpoint["model_state_dict"]
)

diffusion_model.eval()


energy_model = get_model("Energy").to(assignment4_device)

energy_checkpoint = torch.load(
    "checkpoints/energy/energy_cifar10.pth",
    map_location=assignment4_device,
)

energy_model.load_state_dict(
    energy_checkpoint["model_state_dict"]
)

energy_model.eval()

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        start_word=request.start_word,
        num_words=request.length
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    if embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not available in this Docker environment."
        )

    vector = embedding_model.get_embedding(request.word)

    return {
        "word": request.word,
        "has_vector": embedding_model.has_vector(request.word),
        "embedding_length": len(vector),
        "embedding": vector
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    if embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not available in this Docker environment."
        )

    similarity = embedding_model.get_similarity(
        request.word1,
        request.word2
    )

    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": similarity
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "error": "Please upload a valid image file."
        }

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = (
        cnn_transform(image)
        .unsqueeze(0)
        .to(cnn_device)
    )

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

    class_index = predicted_index.item()

    return {
        "filename": file.filename,
        "predicted_class": CLASS_NAMES[class_index],
        "class_index": class_index,
        "confidence": confidence.item(),
    }

@app.get("/generate-digit")
def generate_digit():
    z_dim = 100

    with torch.no_grad():
        noise = torch.randn(1, z_dim, device=mnist_gan_device)
        fake_image = mnist_gan_model.generator(noise)

    # Convert from [-1, 1] to [0, 1]
    fake_image = (fake_image + 1) / 2

    # Shape: [1, 1, 28, 28] -> [28, 28]
    image_array = fake_image.squeeze().cpu().numpy()

    # Convert to PNG image
    image_array = (image_array * 255).clip(0, 255).astype("uint8")
    image = Image.fromarray(image_array, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    from fastapi.responses import StreamingResponse

    return StreamingResponse(buffer, media_type="image/png")

def rgb_tensor_to_response(image_tensor):
    image_tensor = image_tensor.detach().cpu()

    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]

    image_array = (
        image_tensor.permute(1, 2, 0)
        .numpy()
    )

    image_array = (
        image_array * 255
    ).clip(0, 255).astype("uint8")

    image = Image.fromarray(
        image_array,
        mode="RGB",
    )

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
    )

@app.get("/generate-diffusion")
def generate_diffusion():

    with torch.no_grad():

        images = diffusion_model.generate(
            num_images=1,
            diffusion_steps=20,
            image_size=32,
        )

    return rgb_tensor_to_response(images[0])

@app.get("/generate-energy")
def generate_energy():

    images = generate_energy_samples(
        model=energy_model,
        device=assignment4_device,
        num_samples=1,
        image_size=32,
        sampling_steps=60,
        save_path="api_energy.png",
    )

    return rgb_tensor_to_response(images[0])

@app.get("/generate-rl")
def generate_rl(question: str = "What is machine learning?"):
    # Step 1: Generate the answer content
    answer_prompt = f"Question: {question}\nAnswer:"

    answer_inputs = assignment5_tokenizer(
        answer_prompt,
        return_tensors="pt",
    ).to(assignment5_device)

    with torch.no_grad():
        generated_ids = assignment5_model.generate(
            **answer_inputs,
            max_new_tokens=30,
            do_sample=True,
            top_k=40,
            temperature=0.8,
            pad_token_id=assignment5_tokenizer.eos_token_id,
        )

    generated_text = assignment5_tokenizer.decode(
        generated_ids[
            0,
            answer_inputs["input_ids"].shape[1]:
        ],
        skip_special_tokens=True,
    ).strip()

    # Step 2: Ask the RL post-trained model which format to use
    format_prompt = (
        f"Question: {question}\n"
        f"Choose response format:"
    )

    format_inputs = assignment5_tokenizer(
        format_prompt,
        return_tensors="pt",
    ).to(assignment5_device)

    with torch.no_grad():
        format_outputs = assignment5_model(
            **format_inputs
        )

    logits = format_outputs.logits[
        0,
        -1,
        assignment5_action_token_ids,
    ]

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    best_index = torch.argmax(
        probabilities
    ).item()

    selected_format = ASSIGNMENT5_FORMAT_LABELS[
        best_index
    ]

    # Step 3: Apply the selected response format
    response = (
        ASSIGNMENT5_FORMATS[selected_format]["start"]
        + generated_text
        + ASSIGNMENT5_FORMATS[selected_format]["end"]
    )

    return {
        "question": question,
        "response": response,
        "selected_format": selected_format,
        "format_success": selected_format == "A",
        "probabilities": {
            "A": probabilities[0].item(),
            "B": probabilities[1].item(),
            "C": probabilities[2].item(),
        },
    }