### Movie Recommendation Chatbot Using NLTK

This project is a simple chatbot that classifies user input to suggest a movie or respond to queries based on the input it receives. The chatbot leverages **Natural Language Processing (NLP)** techniques using the **Natural Language Toolkit (NLTK)** to process, classify, and respond.

### How it Works:

1. **Installing and Importing Libraries**:
   - Install the NLTK library using `pip install nltk`, and download the necessary datasets and taggers with `nltk.download("popular")`.
   - Import essential libraries for tokenization, stemming, lemmatization, stopword removal, classification, and more.

2. **Preprocessing Text**:
   - The function `preprocess(sentence)` converts the input text to lowercase, tokenizes it, removes stopwords, and returns filtered words.
   - This cleaned data is essential for extracting meaningful features from the text.

3. **Feature Extraction**:
   - The function `extract_feature(text)` processes user input, tags parts of speech (POS), and stems and lemmatizes the words. 
   - The POS tagging helps identify and filter specific types of words (nouns, verbs, etc.) to extract key features for classification.

4. **Dataset Preparation**:
   - A dataset (stored in a CSV file) is loaded using the `get_content(filename)` function. It reads the input text, classifies it into a category, and associates an appropriate response.
   - The function `extract_feature_from_doc(data)` processes this dataset, extracting features from the text and creating a dictionary of categorized responses.

5. **Training the Classifier**:
   - Two classifiers are trained using different algorithms:
     - **Decision Tree Classifier**: `train_using_decision_tree(training_data, test_data)` uses a decision tree model to classify the input and provides accuracy scores on both training and test data.
     - **Naive Bayes Classifier**: `train_using_naive_bayes(training_data, test_data)` trains a Naive Bayes classifier and also calculates accuracy for training and test sets.
   - Accuracy results are printed to give an idea of how well the model is performing on the given data.

6. **Movie Recommendation Logic**:
   - The function `reply(input_sentence)` classifies the user’s input and returns a response based on the preprocessed dataset.
   - If the input matches a category in the dataset, a corresponding response (recommendation or answer) is provided. Otherwise, a fallback message is returned.

7. **Main Chat Loop**:
   - The program runs in a loop, continuously asking the user for input and returning responses until terminated.

### Key Features:

- **Text Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization are performed to clean and normalize user input.
- **Feature Extraction**: POS tagging is used to identify relevant words for classification.
- **Machine Learning Classifiers**: Two classifiers (Decision Tree and Naive Bayes) are trained to classify input into categories.
- **Interactive Chatbot**: The chatbot is designed to provide movie recommendations based on user input, running interactively in the console.


### Dependencies:
- Python 3.x
- NLTK
- Pandas
- NumPy

### How to Run:

1. Install the required libraries using `pip install -r requirements.txt`.
2. Download the necessary NLTK resources using the following commands in Python:
   ```python
   import nltk
   nltk.download("popular")
   ```
3. Prepare the CSV dataset with user inputs, categories, and responses.
4. Run the chatbot with the command:
   ```bash
   python chatbot.py
   ```

### Files in the Repository:
- **chatbot.py**: The main Python file with the chatbot implementation.
- **chatbot.csv**: The dataset containing inputs, categories, and responses.
- **training_data.npy & test_data.npy**: Saved training and testing datasets.

---
