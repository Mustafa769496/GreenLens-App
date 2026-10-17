
# GreenLens 🌿

GreenLens is a plant disease detection and care assistance application built with Python, PyTorch, and Streamlit.

## Features

- **Disease Detection:** Predicts the plant disease class from an uploaded image.
- **Similar Plant Images:** Retrieves visually similar images from the local dataset.
- **Care Recommendations:** Provides care suggestions based on the predicted plant condition.
- **Model Evaluation:** Includes scripts for evaluating disease classification and image similarity.

## Project Structure

```text
GreenLens/
├── app.py
├── disease_detection.py
├── similarity.py
├── recommendations.py
├── evaluate_disease.py
├── evaluate_similarity.py
├── requirements.txt
├── README.md
├── dataset/
└── models/
    ├── plant_disease_model.pth
    └── similarity_index_test.pt
```

## Requirements

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow
- NumPy

## Installation

1. Clone or download this project.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment (recommended).

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Model Files

The disease detection model is stored in:

`models/plant_disease_model.pth`

The similarity feature uses image embeddings generated from the local dataset. The required dataset and any generated similarity index must be available in the expected paths.

## Evaluation

Run the disease detection evaluation script:

```bash
python evaluate_disease.py
```

Run the image similarity evaluation script:

```bash
python evaluate_similarity.py
```

Check each script's configuration and dataset paths before running it.

## Important Notes

- Prediction results are model estimates, not a guaranteed diagnosis.
- Similarity results are based on visual resemblance and do not guarantee that two images show the same disease.
- Model performance may vary on images that were not included in the evaluation dataset.
- Dataset images and trained model weights may need to be provided separately when sharing the project.

## Technologies

- Python
- Streamlit
- PyTorch
- Torchvision
