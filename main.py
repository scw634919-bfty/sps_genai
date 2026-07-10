import io

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from helper_lib.model import get_model

app = FastAPI()

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