# Email/SMS Spam Classifier

A machine learning project that classifies SMS/Email messages as **spam** or **ham (legitimate)**. The notebook explores the dataset, preprocesses message text, visualizes word patterns, converts text into TF-IDF features, compares multiple classifiers, and saves a Multinomial Naive Bayes model with its TF-IDF vectorizer.

## Project Overview

Spam messages can contain unsolicited promotions, suspicious offers, or misleading requests. This project uses supervised machine learning to identify patterns in message text and predict whether a message is spam or legitimate.

## Dataset

- **File:** `spam.csv`
- **Rows:** 5,572 messages
- **Original columns:** `v1`, `v2`, and three mostly empty extra columns
- **Label column:** `v1` (`ham` or `spam`)
- **Message column:** `v2`

The notebook keeps the label and message columns, renames them to `label` and `text`, encodes the labels numerically, checks for duplicate rows, and removes duplicates. The dataset is not included in this README; include `spam.csv` in the repository only if its source and license permit redistribution.

## Workflow

1. **Data loading and preparation**
   - Load `spam.csv` using Latin-1 encoding.
   - Keep the label and message columns.
   - Encode the labels with `LabelEncoder`.
   - Check and remove duplicate records.

2. **Exploratory data analysis**
   - Examine the distribution of ham and spam messages.
   - Explore character, word, and sentence counts.
   - Visualize frequent terms and word clouds for both classes.

3. **Text preprocessing**
   - Convert text to lowercase.
   - Tokenize messages using NLTK.
   - Keep alphanumeric tokens.
   - Remove English stopwords and punctuation.
   - Apply Porter stemming.
   - Join the processed tokens back into a text string.

4. **Feature extraction**
   - Use `TfidfVectorizer` with `max_features=3000`.
   - Split the data into training and test sets using an 80/20 split, `random_state=4`, and stratification.
   - Fit TF-IDF on the training text and transform the test text.

5. **Model comparison**
   - Train and compare 11 classification algorithms.
   - Evaluate each model using accuracy and precision.

6. **Model export**
   - Save the fitted TF-IDF vectorizer to `vectorizer.pkl`.
   - Save the Multinomial Naive Bayes classifier to `model.pkl`.

## Algorithms Compared

- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KN)
- Multinomial Naive Bayes (NB)
- Decision Tree (DT)
- Logistic Regression (LR)
- Random Forest (RF)
- AdaBoost
- Bagging Classifier (BGC)
- Extra Trees Classifier (ETC)
- Gradient Boosting (GBDT)
- XGBoost (XGB)

## Results

The following figures are the results recorded in the notebook's model-comparison table.

| Algorithm | Accuracy | Precision |
|---|---:|---:|
| Extra Trees Classifier (ETC) | 97.58% | 99.07% |
| Support Vector Classifier (SVC) | 97.00% | 98.08% |
| Random Forest (RF) | 96.71% | 98.99% |
| Multinomial Naive Bayes (NB) | 96.91% | 100.00% |
| XGBoost (XGB) | 96.33% | 96.04% |
| Logistic Regression (LR) | 95.45% | 96.67% |
| Gradient Boosting (GBDT) | 94.97% | 96.47% |
| Bagging Classifier (BGC) | 95.84% | 89.29% |
| Decision Tree (DT) | 93.23% | 86.75% |
| AdaBoost | 92.65% | 83.13% |
| K-Nearest Neighbors (KN) | 91.30% | 97.67% |

**Best recorded accuracy:** Extra Trees Classifier — 97.58%.

**Best recorded precision:** Multinomial Naive Bayes — 100%.

The notebook exports the **Multinomial Naive Bayes** model and TF-IDF vectorizer. Accuracy and precision alone do not show every type of error, so recall, F1-score, and a confusion matrix would be useful additions before making claims about real-world performance.

## Repository Structure

```text
email-sms-spam-classifier/
├── email_spam_classifier(1).ipynb
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── README.md
├── requirements.txt
├── app.py                 # optional: add if using the Streamlit app
└── screenshots/           # optional: add screenshots of results or the app
```

You may rename `email_spam_classifier(1).ipynb` to `email_spam_classifier.ipynb` for a cleaner filename. If you rename it, update this tree accordingly.

## Installation

Use Python 3 and create a virtual environment in the project directory.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies used by the notebook:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud xgboost jupyter
```

If you use the optional Streamlit interface, install Streamlit too:

```bash
pip install streamlit
```

## Run the Notebook

Start Jupyter from the repository folder:

```bash
jupyter notebook
```

Open the notebook and run its cells in order. The notebook downloads/uses NLTK tokenization and English stopword resources, so make sure those resources are available in your Python environment.

## Using the Saved Model

The exported files are:
- `vectorizer.pkl` — fitted TF-IDF vectorizer
- `model.pkl` — trained Multinomial Naive Bayes classifier

The notebook trains the vectorizer on **preprocessed text**, so new messages must go through the same `transform_text()` preprocessing function before being passed to `vectorizer.transform()`. Then pass the resulting sparse feature matrix to `model.predict()`.

Example prediction flow:

```python
message = "Congratulations! You have won a free prize. Call now!"
processed_message = transform_text(message)  # use the same preprocessing as the notebook
features = vectorizer.transform([processed_message])
prediction = model.predict(features)[0]

# In this notebook's LabelEncoder, ham is encoded as 0 and spam as 1.
print("Spam" if prediction == 1 else "Not Spam")
```

This snippet assumes `transform_text`, `vectorizer`, and `model` have already been defined or loaded, and that the required NLTK resources are installed. The model files should only be loaded from a trusted source because pickle files can execute code when loaded.

## Limitations and Future Improvements

- Add recall, F1-score, and a confusion matrix to the model comparison.
- Inspect false positives and false negatives.
- Use cross-validation and hyperparameter tuning.
- Test on additional, newer messages to assess generalization.
- Make preprocessing robust and consistent between training and inference.
- If publishing a Streamlit app, document its setup and provide screenshots.
- Check the dataset's license and source before redistributing it.

## Disclaimer

This project is intended for educational purposes. Predictions can be wrong; do not treat a message as safe solely because this model labels it as ham.
