import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

load_dotenv()  ## load all our environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input):
    """
    Gets a response from the Gemini model.

    Args:
        input: The input prompt for the model.

    Returns:
        The text response from the model.
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file):
    """
    Extracts text from a PDF file.

    Args:
        uploaded_file: The uploaded PDF file.

    Returns:
        The extracted text from the PDF.
    """
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Prompt Template
input_prompt = """
Hey Act Like a skilled or very experienced ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving the resumes. Assign the percentage Matching based 
on JD and
the missing keywords with high accuracy
resume:{text}
description:{jd}

I want the response in one single string having the structure
{{"JD Match":"%","MissingKeywords:[]","Profile Summary":""}}
"""

# Streamlit app
st.title("Gspann Smart ATS")
st.text("Check resume whether Profile is fit for job or not using ATS System")
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        response = get_gemini_response(input_prompt.format(text=text, jd=jd))
        st.subheader(response)

def analyze_resume(resume, jd):
    """
    Analyzes a resume against a job description, providing a match percentage, 
    missing keywords, and profile summary suggestions.

    Args:
        resume: The text content of the resume.
        jd: The text content of the job description.

    Returns:
        A dictionary containing the analysis results:
            - 'jd_match': The percentage match between the resume and JD.
            - 'missing_keywords': A list of missing keywords.
            - 'profile_summary': A suggested profile summary.
    """
    # Preprocess text
    stop_words = set(stopwords.words('english'))
    def preprocess_text(text):
        text = re.sub(r'[^\w\s]', '', text).lower()  # Remove punctuation
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if not w in stop_words]
        return ' '.join(filtered_tokens)

    resume_processed = preprocess_text(resume)
    jd_processed = preprocess_text(jd)

    # Calculate TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=500)  # Adjust max_features as needed
    tfidf_matrix = tfidf_vectorizer.fit_transform([resume_processed, jd_processed])

    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    jd_match = round(cosine_sim * 100, 2)

    # Find missing keywords
    jd_keywords = set(jd_processed.split())
    resume_keywords = set(resume_processed.split())
    missing_keywords = list(jd_keywords - resume_keywords)

    # Generate profile summary (optional)
    # You can use another language model or a rule-based system to generate a 
    # concise and impactful profile summary based on the resume and JD.

    return {
        'jd_match': jd_match,
        'missing_keywords': missing_keywords,
        'profile_summary': ''  # Placeholder for profile summary generation
    }