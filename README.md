\# Explainable AI-Based Phishing Email Detection and Cybersecurity Awareness System



\### MSc Computing Dissertation Project



A machine learning-based phishing email detection application developed as part of my MSc Computing dissertation at the University of East London.



\## Project Overview



Phishing emails are a common cybersecurity threat. This project explores how Natural Language Processing (NLP), machine learning and Explainable AI can be combined to classify suspicious emails and provide understandable explanations for model predictions.



The application is developed using Python and Streamlit, with a focus on email classification, model evaluation, interpretability and cybersecurity awareness.



\## Key Features



\- \*\*Email Classification:\*\* Classifies input email text as phishing or safe.

\- \*\*Explainable AI:\*\* Uses LIME to explain individual model predictions.

\- \*\*Batch Email Checker:\*\* Supports classification of multiple email records.

\- \*\*Dataset Overview:\*\* Presents dataset information and class distribution.

\- \*\*Model Evaluation:\*\* Displays classification metrics and model performance comparisons.

\- \*\*Cybersecurity Awareness:\*\* Provides guidance on recognising suspicious emails.



\## Machine Learning Approach



The project follows these main stages:



1\. Dataset preparation and data cleaning

2\. Email text preprocessing

3\. TF-IDF feature extraction

4\. Machine learning model training

5\. Model performance evaluation

6\. Explainable AI integration

7\. Streamlit dashboard development



Four machine learning algorithms were compared:



\- Naive Bayes

\- Logistic Regression

\- Support Vector Machine (SVM)

\- Random Forest



The Support Vector Machine achieved the highest reported test accuracy of \*\*97.98%\*\* in the dissertation evaluation.



This result is dataset-specific and does not guarantee equivalent performance on real-world emails.



\## Technologies Used



\- Python

\- pandas and NumPy

\- scikit-learn

\- Natural Language Processing

\- TF-IDF

\- LIME

\- Streamlit

\- Matplotlib and other evaluation visualisations



\## Project Structure



```text

explainable-ai-phishing-email-detection/

├── app.py

├── README.md

├── requirements.txt

├── .gitignore

├── data/

├── models/

│   ├── best\_phishing\_model.pkl

│   └── tfidf\_vectorizer.pkl

├── notebooks/

│   └── 01\_data\_cleaning.ipynb

├── results/

│   ├── model\_comparison\_results.csv

│   ├── svm\_confusion\_matrix.csv

│   └── dissertation\_graphs/

└── docs/

```



\## How to Run Locally



Create and activate a Python virtual environment:



```bash

python -m venv .venv

```



On Windows:



```bat

.venv\\Scripts\\activate

```



Install the required packages:



```bat

python -m pip install -r requirements.txt

```



Start the Streamlit application:



```bat

python -m streamlit run app.py

```



Then open `http://localhost:8501` in your browser.



\*\*Dataset requirement:\*\* The application expects `data/cleaned\_phishing\_email\_dataset.csv`. The full dataset is not included in the public repository. Users need to obtain the appropriately licensed source dataset and reproduce the preprocessing steps before running dataset-dependent features.



\## Model Evaluation



Model comparison results, a confusion matrix and dissertation graphs are included in the `results/` directory.



The saved classifier and vectorizer were created using scikit-learn 1.6.1. Using a matching environment is recommended when loading these artifacts. Only load pickle files from trusted sources.



\## Limitations



\- Model predictions can produce false positives and false negatives.

\- LIME explanations describe model behaviour, not definitive evidence that an email is malicious.

\- Results depend on the dataset and evaluation conditions.

\- The application is a research prototype rather than a production security service.



\## Future Improvements



Potential extensions include improved generalisation testing, a larger and more diverse evaluation dataset, additional model comparisons and deployment-focused security improvements.



\## Project Purpose



This dissertation demonstrates practical experience in Python, NLP, machine learning, model evaluation, explainable AI and interactive dashboard development.



\*\*Disclaimer:\*\* This project is intended for academic research and educational demonstration. It should not be used as the sole method for determining whether an email is safe.

