# AI-DR-ASSISTANT-Project
# 🤖 AI-DR-ASSISTANT-Project

## Project Overview

The **AI-DR-ASSISTANT-Project** is a deep learning initiative focused on developing an Artificial Intelligence-powered system to assist in the diagnosis of **Diabetic Retinopathy (DR)**.

The goal of this project is to create an accessible and explainable tool for medical professionals by providing a web interface where retinal images can be uploaded, processed by a trained model, and returned with a predicted diagnostic result.

## ✨ Key Features

* **Diabetic Retinopathy (DR) Classification:** Utilizes deep learning models for accurate grading and classification of DR severity from retinal images.
* **Model Explainability (Grad-CAM):** Implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visually highlight the specific regions of the image the model focused on for its prediction. This increases trust and interpretability in the diagnosis. 
* **Web Interface:** Features a user-friendly front-end built with HTML and CSS for easy image submission and result retrieval.
* **Modular Codebase:** Training, utility functions, and model definitions are separated into distinct modules (`Model`, `Functions`, `Utils`) for streamlined development and maintenance.

## 📁 Repository Structure

The core files and directories are organized as follows:

| Directory/File | Description |
| :--- | :--- |
| `Model/` | Contains the trained model files and architecture definitions. |
| `Functions/` | Utility Python scripts for data handling and custom operations. |
| `AI_web/`, `New_Web/` | Source code for the web interface (HTML, CSS, and related files). |
| `Training-*.ipynb` | Jupyter Notebooks detailing the data loading, pre-processing, and model training workflow. |
| `Gradcam.py` | Implementation of the Grad-CAM technique for model visualization. |
| `CopiedFirst1000Rows.csv` | Sample or subset of the dataset used for training/testing. |
| `LICENSE` | The full text of the Apache License, Version 2.0. |

## 🚀 Getting Started

### Prerequisites

You will need the following installed:

* Python (3.x recommended)
* A suitable deep learning framework (likely TensorFlow/Keras or PyTorch, based on file extensions)
* Standard scientific computing libraries (e.g., NumPy, Pandas)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/hudaib2009/AI-DR-ASSISTANT-Project.git](https://github.com/hudaib2009/AI-DR-ASSISTANT-Project.git)
    cd AI-DR-ASSISTANT-Project
    ```
2.  Install the required dependencies (A `requirements.txt` file is recommended to be created for this step):
    ```bash
    pip install -r requirements.txt 
    # Replace with the actual command once a requirements file is added.
    ```

## 📜 License

This project is licensed under the **Apache License, Version 2.0** - see the [LICENSE](LICENSE) file for details.

---
*Developed by the JMI Team.*
