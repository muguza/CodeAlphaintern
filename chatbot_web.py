import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Hospital FAQ Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .chatbot-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        font-weight: 500;
    }
    .bot-message {
        background: linear-gradient(135deg, #00d79f 0%, #009b6f 100%);
        color: white;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(0, 215, 159, 0.3);
        font-weight: 500;
    }
    .stButton button {
        background-color: #667eea !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background-color: #764ba2 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    .stTextInput input {
        border-radius: 8px !important;
        border: 2px solid #667eea !important;
        padding: 12px !important;
    }
    .stSlider {
        padding: 10px !important;
    }
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
    }
    .stInfo {
        background-color: #e8f4f8 !important;
        border-color: #00d79f !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_prepare_chatbot():
    """Load CSV and prepare vectorizer/matrix."""
    df = pd.read_csv("hospital_faqs.csv", on_bad_lines='skip')
    df = df.dropna()
    
    def clean_text(s):
        if not isinstance(s, str):
            s = str(s)
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    # Prepare TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    questions = df['question'].astype(str).apply(clean_text).tolist()
    X = vectorizer.fit_transform(questions)
    
    return df, vectorizer, X, clean_text

def get_chatbot_response(user_input, df, vectorizer, X, clean_text, threshold=0.2):
    """Get chatbot response for user input."""
    user_input_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)
    score = similarity[0, index]
    
    if score < threshold:
        return "Sorry, I don't know the exact answer. Please contact the hospital or try a different question.", score
    
    return df['answer'].iloc[index], score

# Main UI
st.markdown("<h1 style='text-align: center; color: #667eea;'> Hospital FAQ Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 16px;'>Get instant answers to your hospital and health questions</p>", unsafe_allow_html=True)
st.markdown("---")

# Load chatbot components
df, vectorizer, X, clean_text = load_and_prepare_chatbot()

# Sidebar info
with st.sidebar:
    st.header(" About")
    st.info(f"**Total FAQs:** {len(df)}\n\n**Status:** Active\n\nAsk any question about hospital services, health issues, or medical procedures.")
    st.markdown("---")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05,
                                      help="Lower = more answers, Higher = stricter matching")

# Main chat interface
st.markdown("### Ask a Question")

# Input and button in columns
col1, col2 = st.columns([4, 1])
with col1:
    user_question = st.text_input("You:", placeholder="e.g., What are the visiting hours?", key="user_input")
with col2:
    submit_button = st.button("Send", use_container_width=True)

# Display response
if submit_button and user_question.strip():
    response, confidence = get_chatbot_response(user_question, df, vectorizer, X, clean_text, confidence_threshold)
    
    st.markdown(f"<div class='user-message'><b>You:</b> {user_question}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-message'><b>Bot:</b> {response}<br><small>Confidence: {confidence:.2%}</small></div>", unsafe_allow_html=True)

# Display sample questions
st.markdown("---")
st.markdown("### Sample Questions")
sample_questions = [
    "What are the visiting hours at the hospital?",
    "How do I schedule an appointment?",
    "Are emergency services available 24/7?",
    "What should I bring for my first visit?",
    "How can I prevent diabetes?",
]

cols = st.columns(2)
for idx, q in enumerate(sample_questions):
    with cols[idx % 2]:
        if st.button(q, key=f"sample_{idx}", use_container_width=True):
            response, confidence = get_chatbot_response(q, df, vectorizer, X, clean_text, confidence_threshold)
            st.markdown(f"<div class='user-message'><b>You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-message'><b>Bot:</b> {response}<br><small>Confidence: {confidence:.2%}</small></div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; text-align: center;'><p style='color: white; font-weight: bold; margin: 0;'>⚠️ For urgent medical emergencies, call your local emergency services or visit the nearest ER immediately.</p></div>", unsafe_allow_html=True)
