import streamlit as st
import nltk
from nltk import sent_tokenize, pos_tag
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from string import punctuation

# -------------------------
# Ensure NLTK Data is Downloaded (cached after first run)
# -------------------------
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('sentiwordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    return True

download_nltk_data()

# -------------------------
# Helper Functions
# -------------------------
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment_score(text):
    total_score = 0
    raw_sentences = sent_tokenize(text)

    for sentence in raw_sentences:
        sent_score = 0     
        sentence = str(sentence)
        sentence = sentence.replace("<br />"," ").translate(str.maketrans('', '', punctuation)).lower()
        tokens = TreebankWordTokenizer().tokenize(sentence)
        tags = pos_tag(tokens)

        for word, tag in tags:
            wn_tag = penn_to_wn(tag)
            if not wn_tag:
                continue
            lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            synset = synsets[0]
            try:
                swn_synset = swn.senti_synset(synset.name())
                sent_score += swn_synset.pos_score() - swn_synset.neg_score()
            except:
                continue

        if len(tokens) > 0:
            total_score += (sent_score / len(tokens))

    return (total_score / len(raw_sentences)) * 100 if len(raw_sentences) > 0 else 0

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")

st.title("📊 Sentiment Analysis App")
st.caption("By Lakshmi Narasimha Reddy Avula")

st.write("Enter a sentence or paragraph and get the sentiment score!")

user_input = st.text_area("✍️ Your text here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        score = get_sentiment_score(user_input)
        if score > 0:
            st.success(f"😊 Positive Sentiment (Score: {score:.2f})")
        elif score < 0:
            st.error(f"😡 Negative Sentiment (Score: {score:.2f})")
        else:
            st.info(f"😐 Neutral Sentiment (Score: {score:.2f})")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
