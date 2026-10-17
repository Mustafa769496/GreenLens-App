
# recommendations.py

RECOMMENDATION_DB = {
    "Apple_Apple_Scab": {
        "plant": "Apple",
        "condition": "Apple Scab",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves and fruits for dark or olive-colored lesions.",
            "Remove fallen infected leaves where practical.",
            "Improve airflow around the tree.",
            "Consult an agricultural specialist if symptoms spread.",
        ],
    },
    "Apple_Black_Rot": {
        "plant": "Apple",
        "condition": "Black Rot",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves, branches, and fruits for dark lesions.",
            "Remove affected plant material where appropriate.",
            "Monitor the tree for spreading symptoms.",
            "Seek local agricultural advice if symptoms worsen.",
        ],
    },
    "Apple_Cedar_Apple_Rust": {
        "plant": "Apple",
        "condition": "Cedar Apple Rust",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves for orange or yellow spots.",
            "Check other leaves for similar symptoms.",
            "Monitor symptom progression.",
            "Consult an agricultural specialist for confirmation.",
        ],
    },
    "Apple_Healthy": {
        "plant": "Apple",
        "condition": "No disease detected by the model",
        "status": "Model classified as healthy",
        "recommendations": [
            "Continue regular plant inspection.",
            "Maintain suitable watering and growing conditions.",
            "Monitor new leaves and fruit for changes.",
        ],
    },
    "Corn_(maize)_Cercospora_Leaf_Spot": {
        "plant": "Corn",
        "condition": "Cercospora Leaf Spot",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves for elongated lesions.",
            "Monitor other plants for similar symptoms.",
            "Reduce prolonged leaf wetness where practical.",
            "Seek agricultural advice if symptoms spread.",
        ],
    },
    "Corn_(maize)_Common_Rust_": {
        "plant": "Corn",
        "condition": "Common Rust",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves for rust-colored pustules.",
            "Check multiple leaves for symptoms.",
            "Monitor the crop for progression.",
            "Consult an agricultural specialist if symptoms are widespread.",
        ],
    },
    "Corn_(maize)_Healthy": {
        "plant": "Corn",
        "condition": "No disease detected by the model",
        "status": "Model classified as healthy",
        "recommendations": [
            "Monitor crop growth and leaf condition.",
            "Maintain suitable irrigation.",
            "Inspect plants regularly for new symptoms.",
        ],
    },
    "Corn_(maize)_Northern_Leaf_Blight": {
        "plant": "Corn",
        "condition": "Northern Leaf Blight",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves for long, elongated lesions.",
            "Check nearby plants for similar symptoms.",
            "Monitor changes in affected leaves.",
            "Seek agricultural advice about management options.",
        ],
    },
    "Tomato_Bacterial_Spot": {
        "plant": "Tomato",
        "condition": "Bacterial Spot",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves and fruits for small dark spots.",
            "Avoid handling wet plants where practical.",
            "Clean gardening tools before using them on other plants.",
            "Seek expert confirmation before choosing a treatment.",
        ],
    },
    "Tomato_Early_Blight": {
        "plant": "Tomato",
        "condition": "Early Blight",
        "status": "Possible disease",
        "recommendations": [
            "Inspect lower leaves for dark spots with concentric rings.",
            "Remove severely affected leaves when appropriate.",
            "Avoid overhead watering where practical.",
            "Monitor whether symptoms spread.",
        ],
    },
    "Tomato_Healthy": {
        "plant": "Tomato",
        "condition": "No disease detected by the model",
        "status": "Model classified as healthy",
        "recommendations": [
            "Inspect leaves, stems, and fruits regularly.",
            "Maintain consistent, suitable watering.",
            "Provide adequate sunlight and airflow.",
            "Monitor new growth for changes.",
        ],
    },
    "Tomato_Late_Blight": {
        "plant": "Tomato",
        "condition": "Late Blight",
        "status": "Possible disease requiring prompt assessment",
        "recommendations": [
            "Inspect leaves and stems for expanding dark lesions.",
            "Avoid moving potentially infected plant material.",
            "Seek prompt agricultural advice because this disease can spread rapidly.",
            "Do not rely on the image prediction alone to choose treatment.",
        ],
    },
    "Tomato_Septoria_Leaf_Spot": {
        "plant": "Tomato",
        "condition": "Septoria Leaf Spot",
        "status": "Possible disease",
        "recommendations": [
            "Inspect leaves for small circular spots with pale centers.",
            "Remove affected lower leaves when appropriate.",
            "Avoid overhead watering where practical.",
            "Monitor for additional spots.",
        ],
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "condition": "Possible Tomato Yellow Leaf Curl Virus",
        "status": "Possible disease requiring confirmation",
        "recommendations": [
            "Inspect for leaf curling, yellowing, and reduced growth.",
            "Check nearby plants for similar symptoms.",
            "Monitor for whiteflies, which can transmit this virus.",
            "Seek agricultural advice to confirm the cause.",
        ],
    },
}


def generate_smart_recommendations(
    class_name,
    confidence,
    confidence_threshold=0.60,
):
    """
    Return plant-care recommendations for the predicted class.

    Confidence may be provided as a fraction (0.95)
    or as a percentage (95).
    """

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    # Support both 0.95 and 95 formats.
    if confidence > 1:
        confidence /= 100.0

    confidence = max(0.0, min(confidence, 1.0))

    if confidence_threshold > 1:
        confidence_threshold /= 100.0

    confidence_threshold = max(
        0.0,
        min(float(confidence_threshold), 1.0),
    )

    if confidence < confidence_threshold:
        return {
            "plant": "Unknown",
            "condition": "Uncertain prediction",
            "status": "Low model confidence",
            "recommendations": [
                "Take a clear photo in good lighting.",
                "Focus on the affected leaf and avoid blurry images.",
                "Try another image showing the symptoms clearly.",
                "Consult an agricultural specialist if symptoms persist.",
            ],
            "disclaimer": (
                "The prediction is uncertain. These suggestions "
                "are not a confirmed diagnosis."
            ),
        }

    info = RECOMMENDATION_DB.get(class_name)

    if info is None:
        return {
            "plant": "Unknown",
            "condition": "Unrecognized class",
            "status": "No matching recommendation found",
            "recommendations": [
                "Check the predicted class name.",
                "Upload a clear image if necessary.",
                "Consult an agricultural specialist if symptoms persist.",
            ],
            "disclaimer": (
                "This class is not available in the recommendation database."
            ),
        }

    return {
        **info,
        "disclaimer": (
            "AI-generated suggestions are not a confirmed diagnosis. "
            "Model confidence is not necessarily a calibrated probability."
        ),
    }
