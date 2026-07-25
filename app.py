import streamlit as st
import pickle
import re
import PyPDF2
from docx import Document

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


# Extract text from PDF
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Extract text from Word file
def extract_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text


st.title("AI Resume Screening System")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)


if uploaded_file is not None:

    if uploaded_file.type == "application/pdf":
        resume_text = extract_pdf(uploaded_file)

    else:
        resume_text = extract_docx(uploaded_file)


    if st.button("Predict Category"):

        cleaned = clean_text(resume_text)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        st.success("Predicted Category: " + prediction)
        