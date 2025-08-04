# Hate Speech Classification (DL + Embedding Layer)
This project focuses on multi-class classification to determine whether they contain hate speech, offensive language, or neither. It uses deep learning techniques such as Conv1D with an embedding layer that is trained on the dataset

## 🗂️ Dataset 
- Source: [Hate Speech Dataset](https://github.com/troyhunterz/hatespeech-classification/tree/main/dataset)
- Classes:
  - `0` - Hate Speech
  - `1` - Offensive Language
  - `2` - Neither

## 📚 Technologies & Libraries
- Python 3.12.4
- pandas, numpy, mlxtend
- scikit-learn
- Tensorflow / Keras
- Custom library: [`preprocess_tr`](https://github.com/troyhunterz/preprocess_tr)
- spacy

## 🏷️ Features
- Text Preprocessing: Cleaning, lowercasing, removing emails, URLs, special characters, etc
- Tokenization: Text is tokenized, and a vocabulary is built for embedding
- Model: CNN with Conv1D, Embedding, MaxPooling, and Dense layers for classification

## 🤖 Model Training
- Embedding Layer: Transforms words into vectors
- Conv1D Layer: Extracts features from text
- MaxPooling1D: Reduces dimensionality
- Dense Layers: Final classification with `softmax` activation for multi-class output
Training is done with the `Adam` optimizer and `categorical_crossentropy` loss function

## 🔍 Evaluation
Model evaluation is done using confusion matrix and classification report (precision, recall, f1-score)

## 🧪 Sample Prediction
```python
x = 'Hello, how are you?'
model.predict(get_encoded(x))
```

## ⚙️ Setup Instructions
```bash
git clone https://github.com/troyhunterz/hatespeech-classification.git
cd hatespeech-classification-dl
```

## 🧾 License
This project is licensed under the MIT License.

## 👤 Author
troyhunterz

email: ann0nfolder@gmail.com