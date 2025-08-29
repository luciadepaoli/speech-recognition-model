# Speech Recognition Project

This project is a solution for a speech recognition task.

## Project Structure


*   **`speech_recognition.ipynb`**: This Jupyter Notebook contains the main solution to the exercise. It covers the data loading, preprocessing, model training, and evaluation.
*   **`speech_recognition_functions.py`**: This file contains all the functions used in the notebook. It is designed to be used for creating a more robust and production-ready solution, for example, in a Docker container.
*   **`app.py`**: This file contains the FastAPI application to serve the trained model and perform inference.
*   **`random_forest_model.joblib`**: This file is the saved trained model.
*   **`cnn_model.h5`**: This file is the CNN model.
*   **`requirements.txt`**: This file is requirements file.
*   **`/dataset`**: This folder contains the dataset used for training and testing the model.
*   **`/old`**: This folder contains the old FastAPI version.


## Getting Started

### Prerequisites

*   Python 3.x
*   Jupyter Notebook or JupyterLab
*   FastAPI
*   Uvicorn
*   And other libraries mentioned in the notebook.

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Jupyter Notebook

To see the model training process, open and run the `speech_recognition.ipynb` notebook.

### FastAPI Application

To run the FastAPI application for inference, run the following command in your terminal:

```bash
uvicorn app:app --reload
```

This will start a local server, and you can access the API documentation at `http://127.0.0.1:8000/docs`.

## Model

The model used is a Random Forest Classifier, which is saved in the `random_forest_model.joblib` file.

## Dataset

The dataset is located in the `/dataset` folder. It is split into `train` and `test` sets.
