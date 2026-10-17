# GreenLens 🌿 — AI-Assisted Plant Disease Detection & Care Workspace

GreenLens is a modern, end-to-end computer vision workspace designed to diagnose plant leaf diseases, deliver actionable agricultural care guidance, and execute visual similarity searches over local leaf image datasets. 

The application is built using **Python**, **PyTorch**, and **Streamlit**. It features dynamic cloud dataset fetching via **Hugging Face Hub** to bypass repository size limits and provide seamless deployment on Streamlit Cloud.

---

## 📸 Key Features

- 🔬 **Disease Detection & Classification:** High-accuracy leaf disease detection powered by custom PyTorch deep learning models (ResNet50 architecture).
- 🖼️ **Visual Similarity Search:** Retrieve visually similar leaf samples from the dataset using feature embeddings for comparative diagnosis.
- 🌱 **Actionable Care Guidance:** Smart recommendations offering practical steps to treat infected plants or maintain healthy ones.
- 🌐 **Full Arabic & English Support (RTL/LTR):** Complete localization support with dynamic UI direction adjustments and term translations for Arabic users.
- 🌙 **Modern Responsive UI:** Polished interface featuring light/dark mode toggles, adaptive layouts, and high-performance image previews.
- ☁️ **Cloud Native Dataset Retrieval:** Downloads dataset thumbnails dynamically on runtime from Hugging Face Hub without requiring full repository storage[cite: 8].
- 📄 **Downloadable Reports:** Instant generation of downloadable plant diagnosis summaries in standard text format.

---

## 📁 Repository Structure

```text
GreenLens-App/
│
├── .gitattributes             # Git attributes configuration
├── .gitignore                # Ignored files and local build directories
├── README.md                 # Project documentation and guide
├── app.py                    # Streamlit web application & user interface
├── disease_detection.py      # Core PyTorch model loader & inference engine
├── similarity.py             # Image feature extractor & Hugging Face dataset downloader
├── recommendations.py        # Smart plant care recommendation generator
├── requirements.txt          # Python dependencies and library versions
│
└── models/                   # Pre-trained neural network weights
    └── plant_disease_model.pth