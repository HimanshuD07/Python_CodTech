"""
Module: NLP Chatbot Implementation
Author: Himanshu
Purpose: Processes natural language inputs and maps user queries to predefined
         intents using tokenization, lemmatization, and cosine similarity matching.
Tech Stack: Python 3 | NLTK | scikit-learn (TF-IDF + Cosine Similarity)
"""

import re
import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Bootstrap NLTK resources on first run ──────────────────────────────────────
nltk.download("punkt",      quiet=True)
nltk.download("stopwords",  quiet=True)
nltk.download("wordnet",    quiet=True)
nltk.download("punkt_tab",  quiet=True)

# ── Module-level singletons (initialised once for performance) ─────────────────
_lemmatizer  = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

def load_intents() -> dict:
    """
    Returns the in-memory intent dictionary.

    Schema
    ------
    {
        "<tag>": {
            "patterns":  [str, ...]   # sample user utterances
            "responses": [str, ...]   # possible system replies
        }
    }
    Each tag represents one conversational intent.
    """
    intents = {
        "greeting": {
            "patterns": [
                "hello", "hi", "hey", "good morning", "good evening",
                "howdy", "what's up", "greetings"
            ],
            "responses": [
                "Hello! How can I assist you today?",
                "Hey there! What can I do for you?",
                "Hi! Great to see you. How can I help?"
            ]
        },
        "farewell": {
            "patterns": [
                "bye", "goodbye", "see you", "see you later", "take care",
                "good night", "cya", "i am leaving"
            ],
            "responses": [
                "Goodbye! Have a wonderful day!",
                "See you later! Take care.",
                "Bye! Feel free to come back anytime."
            ]
        },
        "thanks": {
            "patterns": [
                "thanks", "thank you", "thank you so much", "many thanks",
                "appreciate it", "cheers", "that was helpful"
            ],
            "responses": [
                "You're welcome! 😊",
                "Happy to help!",
                "Anytime! Let me know if you need anything else."
            ]
        },
        "name": {
            "patterns": [
                "what is your name", "who are you", "what should i call you",
                "tell me your name", "what are you called"
            ],
            "responses": [
                "I'm ChatBot, your AI-powered assistant!",
                "You can call me ChatBot. Nice to meet you!",
                "I go by ChatBot — built with Python and NLTK."
            ]
        },
        "capabilities": {
            "patterns": [
                "what can you do", "what are your capabilities",
                "how can you help me", "what do you know",
                "tell me what you can do", "your features"
            ],
            "responses": [
                "I can answer general questions, have a conversation, and assist with basic queries!",
                "I'm capable of understanding your intent and responding meaningfully. Ask me anything!",
                "I understand natural language and can chat with you. Try me!"
            ]
        },
        "age": {
            "patterns": [
                "how old are you", "what is your age", "when were you born",
                "when were you created", "how long have you existed"
            ],
            "responses": [
                "I was just brought to life for this internship project, so I'm pretty new!",
                "Age is just a number — but I was created recently for Task 3!",
                "I'm as young as the latest Python version. 😄"
            ]
        },
        "weather": {
            "patterns": [
                "what is the weather", "how is the weather today",
                "will it rain", "is it sunny", "temperature outside",
                "weather forecast", "current weather"
            ],
            "responses": [
                "I don't have live weather data, but you can check weather.com!",
                "For real-time weather, try asking Google or a weather API!",
                "I can't check the weather right now, but a quick search will tell you!"
            ]
        },
        "joke": {
            "patterns": [
                "tell me a joke", "say something funny", "make me laugh",
                "joke", "humor me", "i need a laugh"
            ],
            "responses": [
                "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
                "Why did the Python developer cross the road? To get to the other side… of the for loop!",
                "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads."
            ]
        },
        "help": {
            "patterns": [
                "help", "i need help", "can you help me", "assist me",
                "support", "i am confused", "i don't understand"
            ],
            "responses": [
                "Of course! Tell me what you need help with.",
                "I'm here to help. What seems to be the issue?",
                "Sure thing! Describe your problem and I'll do my best."
            ]
        },
        "creator": {
            "patterns": [
                "who created you", "who built you", "who made you",
                "who is your developer", "who programmed you"
            ],
            "responses": [
                "I was built by Himanshu as part of an internship project — Task 3!",
                "Himanshu created me using Python and NLTK. 🚀",
                "A CSE student named Himanshu brought me to life!"
            ]
        }
    }
    return intents


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NLP PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# Initialising NLP pipeline: applying tokenization and lemmatization to
# normalize raw string inputs before comparison.
def preprocess_input(text: str) -> str:
    """
    Cleans and normalises a raw user string through a four-stage pipeline:
        1. Lowercasing       — case normalisation
        2. Punctuation strip — regex-based noise removal
        3. Tokenization      — split sentence → word tokens
        4. Stop-word removal + Lemmatization → canonical base forms

    Parameters
    ----------
    text : str
        Raw user input from terminal.

    Returns
    -------
    str
        Space-joined string of lemmatized, filtered tokens.
    """
    # Stage 1 & 2: lowercase + strip non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text.lower())

    # Stage 3: tokenize into individual word units
    tokens = nltk.word_tokenize(text)

    # Stage 4: remove stop-words and lemmatize surviving tokens
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _stop_words and token.strip()
    ]

    return " ".join(tokens)


def preprocess_patterns(intents: dict) -> tuple[list[str], list[str]]:
    """
    Pre-processes all intent patterns into a flat parallel corpus.

    Returns
    -------
    corpus : list[str]
        Cleaned pattern strings aligned with `tags`.
    tags : list[str]
        Intent tag for each pattern in `corpus`.
    """
    corpus, tags = [], []
    for tag, data in intents.items():
        for pattern in data["patterns"]:
            corpus.append(preprocess_input(pattern))
            tags.append(tag)
    return corpus, tags


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INTENT MATCHING (TF-IDF + COSINE SIMILARITY)
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorizer(corpus: list[str]) -> tuple:
    """
    Fits a TF-IDF vectorizer on the pre-processed pattern corpus.

    TF-IDF encodes term importance relative to the entire corpus,
    making similarity scores more semantically meaningful than raw
    keyword overlap.

    Parameters
    ----------
    corpus : list[str]
        Pre-processed intent patterns.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted vectorizer instance.
    corpus_matrix : sparse matrix
        TF-IDF representation of the full corpus.
    """
    vectorizer = TfidfVectorizer()
    corpus_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, corpus_matrix


def determine_intent(
    user_input: str,
    vectorizer: TfidfVectorizer,
    corpus_matrix,
    tags: list[str],
    threshold: float = 0.15
) -> str | None:
    """
    Calculates vector similarity between the user query and each pattern
    to determine the optimal intent mapping.

    Strategy
    --------
    1. Transform the user input into TF-IDF space.
    2. Compute cosine similarity against every corpus pattern vector.
    3. Return the intent tag for the highest-scoring match if it meets
       the confidence threshold; otherwise return None (unknown intent).

    Parameters
    ----------
    user_input : str
        Pre-processed user query string.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (same instance used at build time).
    corpus_matrix : sparse matrix
        TF-IDF matrix of all training patterns.
    tags : list[str]
        Parallel tag list aligned with corpus rows.
    threshold : float
        Minimum cosine similarity score to accept a match.

    Returns
    -------
    str | None
        Matched intent tag or None if confidence is below threshold.
    """
    # Calculating vector similarity to determine the optimal response mapping.
    input_vector    = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_vector, corpus_matrix).flatten()

    best_idx   = similarity_scores.argmax()
    best_score = similarity_scores[best_idx]

    return tags[best_idx] if best_score >= threshold else None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESPONSE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(intent_tag: str | None, intents: dict) -> str:
    """
    Selects a response from the matched intent bucket.

    Randomises over available responses to avoid repetitive outputs.
    Falls back to a generic unknown-intent message if matching failed.

    Parameters
    ----------
    intent_tag : str | None
        Resolved intent tag from `determine_intent()`.
    intents : dict
        Full intent knowledge base.

    Returns
    -------
    str
        A natural language response string.
    """
    if intent_tag is None:
        return (
            "I'm not sure I understood that. Could you rephrase? "
            "(Type 'help' if you need guidance.)"
        )
    return random.choice(intents[intent_tag]["responses"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN APPLICATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_chatbot() -> None:
    """
    Orchestrates the full chatbot pipeline and manages the terminal session.

    Pipeline per turn
    -----------------
    raw input → preprocess → intent match → response generation → display

    Exit commands : 'quit' | 'exit' | 'q'
    """
    print("=" * 60)
    print("  NLP ChatBot  |  Internship Task 3")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)

    # ── One-time setup: build knowledge base and TF-IDF index ─────────────────
    intents              = load_intents()
    corpus, tags         = preprocess_patterns(intents)
    vectorizer, corpus_matrix = build_vectorizer(corpus)

    EXIT_COMMANDS = {"quit", "exit", "q", "bye", "goodbye"}

    # Instantiating continuous terminal-based I/O loop.
    # Listening for designated termination commands.
    while True:
        try:
            raw_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Graceful exit on Ctrl+C or piped EOF
            print("\n\nChatBot: Session terminated. Goodbye! 👋")
            break

        # ── Guard: empty input ────────────────────────────────────────────────
        if not raw_input:
            print("ChatBot: Please type something!")
            continue

        # ── Guard: exit command ───────────────────────────────────────────────
        if raw_input.lower() in EXIT_COMMANDS:
            print("ChatBot: Goodbye! It was great chatting with you. 👋")
            break

        # ── NLP Pipeline ─────────────────────────────────────────────────────
        cleaned_input  = preprocess_input(raw_input)
        intent_tag     = determine_intent(
                             cleaned_input, vectorizer,
                             corpus_matrix, tags
                         )
        response       = generate_response(intent_tag, intents)

        print(f"ChatBot: {response}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_chatbot()
