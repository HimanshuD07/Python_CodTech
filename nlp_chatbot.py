import re
import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pre-download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Initialize helpers
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# --- Knowledge Base ---
# TODO: Move this dictionary to a JSON file for easier updates later
def load_intents():
    return {
        "greeting": {
            "patterns": ["hello", "hi", "hey", "good morning", "howdy"],
            "responses": ["Hello! How can I assist you today?", "Hi! Great to see you."]
        },
        "farewell": {
            "patterns": ["bye", "goodbye", "see you", "take care", "exit"],
            "responses": ["Goodbye! Have a wonderful day!", "See you later!"]
        },
        "thanks": {
            "patterns": ["thanks", "thank you", "appreciate it"],
            "responses": ["You're welcome! 😊", "Happy to help!"]
        },
        "name": {
            "patterns": ["what is your name", "who are you"],
            "responses": ["I'm ChatBot, your AI assistant!", "You can call me ChatBot."]
        },
        "location": {
            "patterns": ["where are you", "where do you live", "where are you from"],
            "responses": ["I live in the cloud, specifically inside this Python environment!", "I'm from the world of code."]
        },
        "creator": {
            "patterns": ["who created you", "who built you", "who made you"],
            "responses": ["I was built by Himanshu for an internship project!", "Himanshu created me using Python and NLTK. 🚀"]
        }
    }

# --- NLP Logic ---
def preprocess_input(text):
    text = re.sub(r"[^a-z\s]", "", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.strip()]
    return " ".join(tokens)

def prepare_engine(intents):
    corpus, tags = [], []
    for tag, data in intents.items():
        for pattern in data["patterns"]:
            corpus.append(preprocess_input(pattern))
            tags.append(tag)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, tags

def run_chatbot():
    print("-" * 30)
    print("  NLP ChatBot | Task 3")
    print("-" * 30)

    intents = load_intents()
    vectorizer, matrix, tags = prepare_engine(intents)

    while True:
        try:
            user_text = input("\nYou: ").strip()
            if user_text.lower() in ["quit", "exit", "q"]:
                print("ChatBot: Goodbye! 👋")
                break

            cleaned = preprocess_input(user_text)
            input_vec = vectorizer.transform([cleaned])
            scores = cosine_similarity(input_vec, matrix).flatten()

            if scores.max() >= 0.15:
                tag = tags[scores.argmax()]
                print(f"ChatBot: {random.choice(intents[tag]['responses'])}")
            else:
                print("ChatBot: I'm not sure I understood that. Could you rephrase?")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    run_chatbot()
