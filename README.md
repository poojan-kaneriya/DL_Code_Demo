# Software Design Principles for Deep Learning Models

## Project Overview
This project demonstrates the practical application of **Software Design Principles** in building a modular, scalable, reproducible, and maintainable deep learning pipeline using a **Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10** dataset.

## Project Structure

```
CODE_DEMO/
├── code_demo.ipynb
├── main.py
├── config/
│   └── config.yaml
├── data/
│   ├── cifar-10-python.tar.gz
│   ├── cifar-10-batches-py/
│   └── dataloader.py
├── models/
│   ├── simple_cnn.py
│   └── __init__.py
├── trainer/
│   ├── train.py
│   ├── evaluate.py
│   └── __init__.py
├── inference/
│   └── predict.py
├── utils/
│   ├── helpers.py
│   └── __init__.py
├── requirements.txt
└── venv/
```

### Folder Descriptions:
- `config`: Stores hyperparameters and configuration settings.
- `data`: Contains dataset files and custom PyTorch data loading scripts.
- `models`: Holds CNN model definitions.
- `trainer`: Includes scripts for training and evaluating the models.
- `inference`: Scripts used for model predictions.
- `utils`: General-purpose utility functions.
- `code_demo.ipynb`: Interactive Jupyter notebook demonstrating full workflow.
- `main.py`: Script orchestrating the entire pipeline.

## Setup Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### Step 2: Create Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

- To run the complete pipeline:

```bash
python main.py
```

- For individual modules:

```bash
# Training
python trainer/train.py

# Evaluation
python trainer/evaluate.py

# Prediction
python inference/predict.py
```

## Using the Jupyter Notebook

To explore interactively:

```bash
jupyter notebook
```

Open `code_demo.ipynb` for a step-by-step demonstration.

## Evaluation Metrics
- **Accuracy**: Overall model performance metric.
- **Cross-Entropy Loss**: Used during training to optimize the model.
- **Confusion Matrix**: Visual representation of model predictions.

## Key Software Design Principles Demonstrated
- **Modularity**: Separate, interchangeable components.
- **Separation of Concerns**: Clear isolation between tasks.
- **Reusability**: Reusable utilities and functions.
- **Reproducibility**: Configuration files, seeds, environment management.
- **Scalability**: Easy extension and modification of the project.
- **Maintainability**: Logical organization, clear documentation, ease of debugging.

## Contributors
- **Hazel Mahajan**
- **Poojan Kaneriya**

---

Thank you for checking out our project!
