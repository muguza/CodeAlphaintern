# hospital FAQ Chatbot

## Project Description

The "hospital FAQ Chatbot" is a Natural Language Processing (NLP) application designed to answer common hospital and health-related questions automatically.

The system uses 'TF-IDF (Term Frequency–Inverse Document Frequency)' vectorization and "Cosine Similarity" to compare user queries with a predefined FAQ dataset and return the most relevant answer.

This project was developed as part of an AI internship to demonstrate practical implementation of NLP techniques in real-world applications.

## Project Objectives

- To build a simple AI-powered FAQ chatbot.
- To apply NLP preprocessing techniques.
- To implement TF-IDF vectorization for text representation.
- To use cosine similarity for intelligent question matching.
- To develop a simple web-based chat interface.

## Features

- Answers medical and hospital-related FAQs
- Text preprocessing (lowercasing & punctuation removal)
- TF-IDF vectorization
- Cosine similarity matching
- Confidence threshold to reduce incorrect answers
- Simple web interface using Streamlit
- Expandable dataset for future improvements

## System Architecture

The chatbot follows this processing pipeline:

1. Load FAQ dataset (CSV file)
2. Clean and normalize text
3. Convert questions into numerical vectors using TF-IDF
4. Compute cosine similarity between user query and dataset questions
5. Select the most similar question
6. Return the corresponding answer

## Technologies Used

- Python 3.
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project Structure

AI-Medical-Chatbot/
│
├── medical_faq.csv        # Dataset containing FAQ questions and answers
├── chatbot_web.py         # Main Streamlit application
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies

## Installation & Setup

### Step 1: Clone the Repository

git clone https://github.com/your-username/chatbot FAQs.git
cd chatbot FAQs

Or download and extract the ZIP file.

### Step 2: Install Dependencies

- Make sure Python 3.x is installed, then run:
- pip install -r requirements.txt

### Step 3: Run the Application

Run this streamlit run app.py in the terminal

The chatbot will automatically open in your browser.

## Example Questions

- What are visiting hours?
- How do I schedule an appointment?
- What are symptoms of malaria?
- What are warning signs of stroke?
- How is COVID-19 treated?
- What should I do before surgery?

## Dataset Description

The dataset (`hospital_faqs.csv`) contains structured question-answer pairs related to:

- Hospital services
- Disease symptoms
- Emergency conditions
- Chronic disease management
- Vaccination information
- General health guidance

The dataset can be expanded to improve chatbot accuracy.

## How Similarity Matching Works

- The dataset questions are converted into numerical vectors using TF-IDF.
- When a user enters a question, it is cleaned and vectorized.
- Cosine similarity measures how close the user query is to stored questions.
- The answer of the most similar question is returned.
- A confidence threshold prevents low-quality matches.

## Limitations

- The chatbot does not generate new answers; it retrieves existing ones.
- Accuracy depends on dataset size and quality.
- Chatbot provide most accurate answers when asked correctly as they appear in the dataset
- It does not replace professional medical consultation.
- Semantic understanding is limited compared to deep learning models.

## Future Improvements

- Use Sentence Transformers for semantic understanding
- Expand dataset to 100+ FAQs
- Add chat history functionality
- Deploy online (Render, Railway, or Heroku)
- Add voice input support
- Improve UI design

## Disclaimer

This chatbot is developed for educational purposes only especially for codeAlpha internship given to me.  
It does not provide medical diagnosis or treatment advice.  
Always consult a qualified healthcare professional for medical concerns.

## Author

Muguza John

## Project name

AI Internship Project
Year: 2026

