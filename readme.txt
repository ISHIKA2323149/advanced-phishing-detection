Advanced Phishing Detection System

Overview
This project is designed to detect phishing attempts in emails and URLs using a multilingual machine learning model based on DistilBERT. The system is capable of classifying inputs as "Legitimate" or "Phishing" across multiple languages.

---

Prerequisites
Ensure the following tools and packages are installed:

1. Python 3.8 or above
2. Required Python Packages:
   - pandas
   - torch
   - transformers
   - datasets
   - scikit-learn
   - nltk
   - matplotlib
   - seaborn

   Install the required packages using:
   ```bash
   pip install pandas torch transformers datasets scikit-learn nltk matplotlib seaborn
   ```

3. NLTK Data:
   Download the necessary NLTK datasets:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4.Hardware Requirements**:
   - GPU (optional but recommended for faster model training and inference)
   - At least 8GB RAM

---

Project Directory Structure
Ensure the project directory contains the following files and folders:

- `distilbert_phishing_model/` - Directory to save the trained model
- `results/` - Directory to save results from the model training
- `templates/` - Additional templates for with a index.html file
- `Hugging_face_transformer.ipynb` - python file for main code
- `app.py` - Main application script for running the model
- `dataset_phishing.csv` - Dataset containing phishing data
- `emails.csv` - Dataset with email samples
- `Phishing_Legitimate_full.csv` - Comprehensive dataset for phishing and legitimate samples
- `readme.txt` - Instructions to run the project

---

Steps to Run the Project

1. Unzip the Project Folder
Unzip the provided folder and navigate to the project directory.

```bash
unzip advanced_phishing_detection.zip
cd advanced_phishing_detection
```

2. Prepare the Datasets
Ensure the datasets (`dataset_phishing.csv`, `emails.csv`, `Phishing_Legitimate_full.csv`) are located in the project directory.

3. Preprocess the Data
Run the preprocessing steps included in the scripts to:
- Remove stopwords
- Tokenize the text
- Concatenate email messages and URLs

Preprocessing is automated in the `app.py` script and Jupyter Notebook.

4. Train the Model
1. Open the Jupyter Notebook `Hugging_face_transformer.ipynb` or run the script in the terminal.
2. Run the training script to fine-tune the DistilBERT model:
   ```bash
   python app.py
   ```
3. Ensure the `trainer.train()` method is uncommented in the code to train the model on the provided datasets.

5. Evaluate the Model
The evaluation includes:
- Calculating metrics such as accuracy, F1 score, precision, and recall.
- Displaying a confusion matrix.
- Generating a multilingual accuracy graph.

Run the evaluation in the Jupyter Notebook or terminal after training.

6. Predict Phishing Attempts
Use the `classify_email` function to classify individual emails and URLs:

```python
from app import classify_email

def example_prediction():
    email = "Your account has been compromised."
    url = "http://fake-url.com"
    result = classify_email(email, url)
    print(result)  # Outputs: 'Phishing' or 'Legitimate'

example_prediction()
```

7. Reproduce Results
To reproduce the results:
1. Run the training script using the provided datasets.
2. Evaluate the trained model on the test dataset.
3. Review the accuracy, F1 score, and other metrics.


---

## Additional Notes
1. Update file paths in the code if the datasets are moved to a different directory.
2. Use GPU acceleration for faster training and evaluation.
3. For issues or questions, review the error logs in the `results/` directory.

---

For further assistance, please contact ishika upadhyay at ishikaupadhyay01@gmail.com

