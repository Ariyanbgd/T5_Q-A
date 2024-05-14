# T5 for Question Answering

This repository contains a TensorFlow implementation of the T5 model for question answering tasks. The model is first pre-trained on the C4 dataset using a masked language model (MLM) task, and then fine-tuned on the Stanford Question Answering Dataset (SQuAD). Finally, the model can be tested with random questions.

## Setup

### Prerequisites

Ensure you have the following libraries installed:

- TensorFlow
- TensorFlow Text
- NumPy
- Termcolor
- Argparse

You can install the required packages using pip:

```bash
pip install tensorflow tensorflow-text numpy termcolor
```

### Directory Structure
Make sure your directory structure looks like this:
```bash
.
├── data
│   ├── c4-en-10k.jsonl            # Pre-training dataset
│   └── train-v2.0.json            # Fine-tuning dataset
├── pretrained_models
│   ├── sentencepiece.model        # Pre-trained tokenizer model
│   ├── model_c4                   # Pre-trained model directory
│   └── model_qa3                  # Fine-tuned model directory
├── main.py                        # Main script
└── transformer_utils.py           # Utility functions and Transformer model implementation
```


## Running the Script
### Pre-training
To pre-train the model on the C4 dataset:
```bash
python main.py --pretrain True
```

### Fine-tuning
To fine-tune the pre-trained model on the SQuAD dataset:
```bash
python main.py --finetune True
```

### Testing
To test the fine-tuned model with a random question:
```bash
python main.py --test-idx 10408
```