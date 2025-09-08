# News-Text-Classification

## Overview

This repository contains a Jupyter Notebook for fine-tuning a BERT model on the AG News dataset for text classification. The project demonstrates how to load a dataset using TensorFlow Datasets, preprocess and tokenize the data with Hugging Face Transformers, create a custom PyTorch dataset, and train a sequence classification model using the `Trainer` API.

The goal is to classify news articles into one of four categories: World, Sports, Business, or Sci/Tech. The notebook uses the `bert-base-uncased` pretrained model and fine-tunes it on the `ag_news_subset` dataset.

## Dataset

- **Source**: AG News subset from TensorFlow Datasets (`ag_news_subset`).
- **Description**: A collection of news headlines categorized into 4 classes.
- **Size**: 120,000 training examples and 7,600 test examples.
- **Classes**: 
  - 0: World
  - 1: Sports
  - 2: Business
  - 3: Sci/Tech

The dataset is loaded directly in the notebook using `tfds.load("ag_news_subset", with_info=True, as_supervised=True)`.

## Requirements

The project requires the following Python libraries:

- `transformers` (Hugging Face)
- `datasets` (Hugging Face)
- `tensorflow_datasets` (TFDS)
- `torch` (PyTorch)
- `tensorflow` (for dataset loading)
- `os` (standard library)

You can install them using:

```bash
pip install transformers datasets tensorflow torch tensorflow_datasets
```

For reproducibility, use Python 3.12+ and ensure you have a GPU available for faster training (mixed precision with `fp16=True` is enabled).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/santhoshkrishnan30/News-Text-Classification.git
   cd News-Text-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # If requirements.txt is added; otherwise, use the command above
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

## Usage

The main file is `Fine_Tune.ipynb`. To run it:

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook Fine_Tune.ipynb
   ```

2. Execute the cells step-by-step:
   - Step 0: Disable Weights & Biases (WandB) logging.
   - Step 1: Load the AG News dataset.
   - Step 2: Tokenize the data using BERT tokenizer.
   - Step 3: Create a custom PyTorch dataset.
   - Step 4: Load the BERT model for sequence classification.
   - Step 5: Set training arguments (e.g., batch size, epochs).
   - Step 6: Initialize and run the Trainer.
   - Step 7: Train the model.
   - Step 8: (Optional) Evaluate and save the model.

After training, the model and tokenizer are saved to `./results` (checkpoints) and can be loaded for inference.

### Example Inference (After Training)

Add this to a new cell or script to test the model:

```python
from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline("text-classification", model="./fine-tuned-bert-agnews", tokenizer="./fine-tuned-bert-agnews")

# Example prediction
result = classifier("AMD's new dual-core Opteron chip is designed for corporate computing.")
print(result)
```

## Training Details

- **Model**: `bert-base-uncased` (110M parameters).
- **Tokenizer**: BERT tokenizer with max_length=64 for efficiency.
- **Batch Size**: 32 (train/eval; adjust based on GPU memory).
- **Epochs**: 1 (for quick demo; increase to 3+ for better results).
- **Optimizer**: AdamW (default in Trainer).
- **Learning Rate**: 5e-5.
- **Mixed Precision**: Enabled (`fp16=True`) for faster training on GPU.
- **Logging**: Every 50 steps; evaluation every 200 steps.
- **Hardware**: Tested on Google Colab with GPU support.

To retrain:
- Run `trainer.train()` in the notebook.
- Monitor loss via printed logs (WandB is disabled).

Expected training time: ~15-20 minutes per epoch on a GPU (e.g., Tesla T4).

## Results

After 1 epoch:
- Training Loss: Decreases from ~0.74 to ~0.20 (as shown in notebook logs).
- Evaluation: Run `trainer.evaluate()` post-training for accuracy/metrics (typically ~90%+ accuracy on test set after full fine-tuning).

For better performance, train for more epochs or increase max_length.

## Contributing

Feel free to fork the repository and submit pull requests for improvements, such as adding metrics computation or hyperparameter tuning.



## Author

- **Santhosh Krishnan**
- GitHub: [santhoshkrishnan30](https://github.com/santhoshkrishnan30)
- Contact: santhoshkrishnan3006@gmail.com

If you have questions or issues, open a GitHub issue!
