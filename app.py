import os
import nltk
import numpy as np
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------
# ENV
# --------------------
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --------------------
# NLTK
# --------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------
# FASTAPI
# --------------------
app = FastAPI(title="VIHA AI Chatbot")

class ChatRequest(BaseModel):
    message: str

# --------------------
# PREPROCESS
# --------------------
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)

# --------------------
# AUTO SYNONYM GENERATOR (1000+ possible)
# --------------------
def expand_with_synonyms(text):
    words = text.split()
    expanded = set(words)

    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))

    return " ".join(expanded)

# --------------------
# FAQ DATA
# --------------------
faq = {
    "what courses are you offering": "We offer Data Science, ML, Full Stack, SAP, Python, and more.",
    "machine learning course fee": "Machine Learning course fee is ₹30,000.",
    "where is institute located": "We are located in Madhapur, Hyderabad.",
    "do you provide placement": "Yes, placement assistance is provided.",
    "what courses are you offering": "We provide training across multiple programming languages and technologies.",
    "i want learn machine learning": "We have a comprehensive ML course from basics to advanced. Want to know the course structure?",
    "what about libraries for machine learning": "We cover all required ML libraries like Pandas, NumPy, Scikit-learn, TensorFlow.",
    "what is the fee structure for machine learning": "The fee is ₹30,000 for the complete Machine Learning course.",
    "where is the institute located": "Our institute is located in Madhapur, Hyderabad.",
    "what kind of projects will we work on": "Projects include predictive modeling, NLP, image recognition, and more.",
    "are you providing certification about this course": "Yes, we provide certification upon course completion.",
    "what is the duration of this course": "The course lasts 5 months with 2 classes per week.",
    "what is the mode of learning": "The course is conducted online.",
    "should i pay the fee in installments": "Yes, we allow payment in 2 installments.",
    "fee structure": "Java Fullstack: ₹40,000 | Data Science: ₹45,000 | HTML, CSS, C++: ₹20,000 | SAP: ₹30,000.",
    "class timings": "Classes are twice a week, each class lasts 2 hours.",
    "how can i enroll": "You can enroll by visiting our website and filling out the registration form.",
    "do you offer online classes": "Yes, we offer online classes through our platform.",
    "is there a weekend batch": "Yes, weekend batches are available.",
    "what is the batch size": "Our typical batch size is 20-25 students.",
    "what is the teaching methodology": "We follow a hands-on approach with real-world projects.",
    "what is the assessment process": "Assessments include quizzes, assignments, and a final project.",
    "what kind of course do you offer": "We offer courses in Data Science, Machine Learning, Web Development, Python Fullstack, Data Analytics, ServiceNow, Salesforce, and more.",
    "what is the fee for each course": "The fee varies by course. Please specify which course you're interested in.",
    "what tools will we use": "Python, Jupyter, TensorFlow, Scikit-learn, and AWS.",
    "do you provide certification": "Yes, certification is provided.",
    "who are the trainers": "Our trainers are industry professionals in Data Science & ML.",
    "do you help with placements": "Yes, placement assistance is provided with resume support.",
    "is there any refund policy": "No, we do not offer refunds.",
    "what projects will we do": "You will work on predictive modeling, NLP, and real-world projects.",
    "is this course suitable for beginners": "Yes, it is beginner-friendly with step-by-step teaching.",
    "what are the prerequisites": "Basic programming and statistics help, but not mandatory.",
    "do you provide study material": "Yes, detailed study material and recordings are provided.",
    "what is the last date for admission": "The last date is the 30th of this month.",
    "can i join without any background in tech": "Yes, the course is open to all backgrounds.",
    "do you offer demo classes": "Yes, free demo classes are available.",
    "do you provide job guarantee": "We offer placement support, but not a job guarantee.",
    "what companies are you partnered with": "We partner with top tech companies for placements.",
    "is interview preparation included": "Yes, mock interviews and guidance are included.",
    "do you provide internships": "Yes, internships are available for selected students.",
    "can i access classes on mobile": "Yes, classes are accessible on mobile.",
    "what is the success rate of placements": "We have 85% placement success rate.",
    "do you help build resumes": "Yes, resume building is included.",
    "what languages are used in training": "Training is conducted in English.",
    "do you provide lifetime access": "You get 1 year access to course materials.",
    "what is your contact number": "Call us at +91-77414787873 for queries.",
    "do you offer courses for school students": "Yes, coding courses for school students are available.",
    "what is the difference between basic and advanced courses": "Basic = foundations, Advanced = deeper practical applications."

}

questions = list(faq.keys())
answers = list(faq.values())

# --------------------
# EMBEDDINGS + FAISS
# --------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(questions)

dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(question_embeddings))

# --------------------
# RAG SEARCH
# --------------------
def rag_search(query, top_k=1):
    emb = model.encode([query])
    distances, idxs = index.search(np.array(emb), top_k)

    confidence = float(1 / (1 + distances[0][0]))
    return answers[idxs[0][0]], confidence

# --------------------
# GROQ LLM
# --------------------
def llm_answer(prompt):
    res = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an institute assistant chatbot."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=300
    )
    return res.choices[0].message.content.strip()

# --------------------
# CHAT ENDPOINT
# --------------------
@app.post("/chat")
def chat(req: ChatRequest):
    raw = req.message
    clean = preprocess(raw)
    expanded = expand_with_synonyms(clean)

    rag_answer, confidence = rag_search(expanded)

    # 🔥 DECISION LOGIC
    if confidence >= 0.55:
        return {
            "reply": rag_answer,
            "confidence": round(confidence, 2),
            "source": "RAG"
        }

    # Fallback to LLM
    llm_resp = llm_answer(raw)
    return {
        "reply": llm_resp,
        "confidence": round(confidence, 2),
        "source": "LLM"
    }

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
