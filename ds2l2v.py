import re
import pandas as pd
import pymorphy3
import stanza
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def get_stopwords(path='stopwords_ua_list.txt'):
    stop_set = set()

    try:
        with open(path, encoding='utf-8') as file:
            data = file.read()

        cleaned = re.sub(r'[^\w\s\-]', ' ', data)

        for token in cleaned.split():
            word = token.strip().lower()
            if word:
                stop_set.add(word)

        print(f"-= Файл {path}: знайдено {len(stop_set)} стоп-слів =-\n")

    except FileNotFoundError:
        print(f"** Не вдалося знайти файл '{path}'. Стоп-слова не застосовано **\n")
    except Exception as err:
        print(f"** Проблема при читанні стоп-слів: {err} **\n")

    return list(stop_set)

def clean_text(text):
    text = str(text).lower()
    text = text.replace("'", "'").replace("ʼ", "'")
    for _ in range(5):
        text = re.sub(r'(\d+)\s+(\d{3})', r'\1\2', text)
    text = re.sub(r"[,;:\.?!\"«»]|(?<=\s)[-–—](?=\s)", " ", text)
    text = re.sub(r"[^\w\s%'\-\%\$\€\₴\£\¥\₽]", '', text)
    text = re.sub(r'\s[-–—]\b', ' ', text)
    text = re.sub(r'\b[-–—]\s', ' ', text)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def lemmatize(df, column):

    def pymorphy3_lemmatize(text):

        morph = pymorphy3.MorphAnalyzer(lang='uk')
        words = text.split()
        text_lemmatized = ""

        for word in words:
            parsed = morph.parse(word)[0]
            lemma = parsed.normal_form
            text_lemmatized += f"{lemma} "

        return text_lemmatized.strip()
    
    def stanza_lemmatize(text):

        nlp = stanza.Pipeline('uk', processors='tokenize,lemma', verbose=False)

        doc = nlp(text)
        text_lemmatized = ""

        for sentence in doc.sentences:
            for word in sentence.words:
                lemma = word.lemma.lower()
                text_lemmatized += f"{lemma} "

        return text_lemmatized.strip()

    def spacy_lemmatize(text):
        nlp = spacy.load('uk_core_news_sm')

        doc = nlp(text)

        text_lemmatized = ""
        for token in doc:
            lemma = token.lemma_
            text_lemmatized += f"{lemma} "

        return text_lemmatized.strip()
    
    def count_unique_lemmas(corpus):
        unique_lemmas = set()
        for text in corpus:
            words = text.split()
            for lemma in words:
                unique_lemmas.add(lemma)
        return len(unique_lemmas)

    print("-= В процесі: лематизація pymorphy3... =-")
    df["pymorphy3"] = df[column].apply(func=pymorphy3_lemmatize)
    print(f"Унікальних лем pymorphy3: {count_unique_lemmas(df["pymorphy3"])}.\n")
    print("-= В процесі: лематизація Stanza... =-")
    df["Stanza"] = df[column].apply(func=stanza_lemmatize)
    print(f"Унікальних лем Stanza: {count_unique_lemmas(df["Stanza"])}.\n")
    print("-= В процесі: лематизація spaCy... =-")
    df["spaCy"] = df[column].apply(func=spacy_lemmatize)
    print(f"Унікальних лем spaCy: {count_unique_lemmas(df["spaCy"])}.\n")

def bow_vectorize(text, filename):
    global data

    print("\n-= В процесі: Bag of Words... =-")
    stopwords = get_stopwords()

    bow_vectorizer = CountVectorizer(stop_words=stopwords)
    bow = bow_vectorizer.fit_transform(text)

    with open(f"{filename}.txt", "w", encoding="utf-8") as f:
        for i, row in enumerate(bow.toarray(), 1):
            row_str = " ".join(map(str, row))
            f.write(f"id={i} -> {row_str}\n")
    
    print(f"Матриця Bag of Words: {bow.shape}")

    print("\n-= Топ-слова для кожного класу (BoW): =-")

    bow_words = bow_vectorizer.get_feature_names_out()
    
    for lbl in data["label"].unique():
        idx = np.where(data["label"] == lbl)[0]
        class_counts = bow[idx].sum(axis=0).A1
        top_idx = np.argsort(class_counts)[-10:][::-1]
        print(f"\nКлас: {lbl}")
        for i in top_idx:
            if class_counts[i] > 0:
                print(bow_words[i], f"({class_counts[i]})")

def tfidf_vectorize(text, filename):
    global data

    print("\n-= В процесі: TF-IDF... =-")
    stopwords = get_stopwords()

    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(text)

    with open(f"{filename}.txt", "w", encoding="utf-8") as f:
        for i, row in enumerate(tfidf.toarray(), 1):
            row_str = " ".join(map(str, row))
            f.write(f"id={i} -> {row_str}\n")
    
    print(f"Матриця TF-IDF: {tfidf.shape}")

    print("\n-= Топ-слова для кожного класу (TF-IDF): =-")

    tfidf_words = tfidf_vectorizer.get_feature_names_out()
    
    for lbl in data["label"].unique():
        idx = np.where(data["label"] == lbl)[0]
        mean_tfidf = tfidf[idx].mean(axis=0).A1
        top_idx = np.argsort(mean_tfidf)[-10:][::-1]
        print(f"\nКлас: {lbl}")
        for i in top_idx:
            if mean_tfidf[i] > 0:
                print(tfidf_words[i], f"({mean_tfidf[i]:.3f})")

if __name__ == "__main__":
    data = pd.read_csv("corpus.csv", sep=";", engine="python", encoding="utf-8")
    data = data.drop(columns=[c for c in data.columns if "Unnamed" in c or c.strip() == ""], errors="ignore")

    data["text"] = data["text"].apply(clean_text)
    data = data[data["text"].str.strip() != ""].reset_index(drop=True)
    print("-= Текст очищено від сторонніх символів =-\n")

    lemmatize(data, "text")
    print(f"\n{"=" * 40}\n{" " * 7}Векторизація лем pymorphy3\n{"=" * 40}")
    bow_vectorize(data["pymorphy3"], "BoW_pymorphy3")
    tfidf_vectorize(data["pymorphy3"], "TFIDF_pymorphy3")
    print(f"\n{"=" * 40}\n{" " * 8}Векторизація лем Stanza\n{"=" * 40}")
    bow_vectorize(data["Stanza"], "BoW_Stanza")
    tfidf_vectorize(data["Stanza"], "TFIDF_Stanza")
    print(f"\n{"=" * 40}\n{" " * 9}Векторизація лем spaCy\n{"=" * 40}")
    bow_vectorize(data["spaCy"], "BoW_spaCy")
    tfidf_vectorize(data["spaCy"], "TFIDF_spaCy")