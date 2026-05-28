# Named Entity Recognition using BiLSTM

This project implements a Named Entity Recognition (NER) pipeline using Bidirectional LSTM (BiLSTM) networks for both POS tagging and entity recognition. The models are trained on the GMB dataset and use masked accuracy to ignore padded tokens during training and evaluation.

---

## Features

* BiLSTM-based POS tagging and NER models
* Custom masked accuracy metric for padded sequences
* Unified inference pipeline for token-level predictions
* Sequential POS + NER prediction workflow
* TensorFlow/Keras implementation

---

## Project Structure

### `Tag.ipynb`

NER preprocessing and training notebook:

* Sentence grouping and tokenization
* Vocabulary and tag indexing
* Sequence padding
* NER model training using BiLSTM

### `POS.ipynb`

POS tagging model notebook:

* BiLSTM-based POS tagger
* Masked accuracy implementation
* Training and evaluation on GMB corpus

### `NER.ipynb`

Final inference pipeline:

* Loads trained POS and NER models
* Accepts raw input sentences
* Returns token-level POS and NER predictions

---

## Model Performance

| Model      | Architecture | Validation Masked Accuracy |
| ---------- | ------------ | -------------------------- |
| POS Tagger | BiLSTM       | **96.4%**                  |
| NER Tagger | BiLSTM       | **95.4%**                  |

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* seqeval

---

## Example

### Input

```python
ner("Apple Inc. released its new iPhone in September 2021 .")
```

### Output

```python
Sentence : Apple Inc. released its new iPhone in September 2021 .

POS : ['NNP', 'NNP', 'VBD', 'PRP$', 'JJ', 'NN', 'IN', 'NNP', 'NNP', '.']

NER : ['B-org', 'I-org', 'O', 'O', 'O', 'O', 'O', 'B-tim', 'I-tim', 'O']
```


## Dataset

The project uses the GMB (Groningen Meaning Bank) dataset containing:

* Sentences
* POS tags
* Named Entity labels

---

## Installation

```bash
pip install tensorflow keras pandas numpy seqeval
```

---

## Run the Project

1. Train the POS model using `POS.ipynb`
2. Train the NER model using `Tag.ipynb`
3. Run `NER.ipynb` for final inference

---

## Future Improvements

* Add CRF layer for improved sequence decoding
* Integrate pre-trained embeddings (GloVe / FastText)
* Deploy as REST API using FastAPI
* Compare performance with transformer-based models

