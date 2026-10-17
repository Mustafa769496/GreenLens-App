import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import models, transforms


# ---------------------------------
# 1. Paths and device
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------
# 2. Original model architecture
# ---------------------------------

class PlantDiseaseClassifier(nn.Module):

    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()

        self.backbone = models.efficientnet_b2(weights=None)

        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, num_features),
            nn.Sigmoid(),
        )

        # Original classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone.features(x)

        pooled = F.adaptive_avg_pool2d(features, 1)
        pooled = torch.flatten(pooled, 1)

        attention_weights = self.attention(features)
        attended_features = pooled * attention_weights

        return self.classifier(attended_features)


# ---------------------------------
# 3. Load local checkpoint
# ---------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    if checkpoint.get("model_architecture") != "PlantDiseaseClassifier":
        raise ValueError("Unexpected model architecture.")

    if len(class_names) != num_classes:
        raise ValueError("Checkpoint class count mismatch.")

    model = PlantDiseaseClassifier(num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    input_size = checkpoint["input_size"]
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    mean = checkpoint.get(
        "normalization_mean",
        [0.485, 0.456, 0.406],
    )
    std = checkpoint.get(
        "normalization_std",
        [0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize(tuple(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, class_names, transform


model, class_names, transform = load_model()


# ---------------------------------
# 4. Disease prediction
# ---------------------------------

def predict_plant(image, top_k=3):
    """
    Predict among the 14 classes supported by this model.

    Args:
        image: PIL Image.
        top_k: Number of predictions to return.

    Returns:
        Dictionary containing the top prediction and alternatives.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    k = min(max(1, int(top_k)), len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=k)

    predictions = []

    for prob, idx in zip(
        top_probs.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        predictions.append({
            "class_name": class_names[idx],
            "confidence": round(float(prob), 4),
        })

    return {
        "predicted_class": predictions[0]["class_name"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
    }


# ---------------------------------
# 5. Basic plant information
# ---------------------------------

PLANT_INFO = {
    "Tomato_Early_Blight": {
        "plant": "Tomato",
        "condition": "Early Blight",
        "description": (
            "A tomato leaf condition commonly associated "
            "with the fungus Alternaria solani."
        ),
        "guidance": [
            "Inspect other leaves for similar symptoms.",
            "Avoid keeping foliage wet for long periods.",
            "Seek local agricultural advice if symptoms spread.",
        ],
    },
    "Tomato_Septoria_Leaf_Spot": {
        "plant": "Tomato",
        "condition": "Septoria Leaf Spot",
        "description": (
            "A tomato leaf disease that can cause small "
            "spots on leaves."
        ),
        "guidance": [
            "Inspect the plant for spreading leaf spots.",
            "Avoid overhead watering when practical.",
            "Seek agricultural advice if symptoms worsen.",
        ],
    },
    "Tomato_Late_Blight": {
        "plant": "Tomato",
        "condition": "Late Blight",
        "description": (
            "A potentially serious tomato disease that can "
            "spread rapidly in suitable conditions."
        ),
        "guidance": [
            "Inspect affected plants and nearby plants.",
            "Avoid moving potentially infected plant material.",
            "Seek prompt agricultural advice if suspected.",
        ],
    },
}


def get_plant_information(class_name):
    return PLANT_INFO.get(
        class_name,
        {
            "plant": "Unknown",
            "condition": "Information unavailable",
            "description": (
                "No specific information is available "
                "for this class yet."
            ),
            "guidance": [
                "Consult a qualified agricultural specialist."
            ],
        },
    )