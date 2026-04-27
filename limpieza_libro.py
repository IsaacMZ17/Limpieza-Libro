import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. Cargar el modelo y el texto
nlp = spacy.load("es_core_news_sm")
with open("libro.txt", "r", encoding="utf-8") as f:
    texto = f.read()

# 2. Preparar el Corpus Lematizado (por oraciones)
doc = nlp(texto)
corpus_lematizado = []

for oracion in doc.sents:
    lemas = [
        token.lemma_.lower() 
        for token in oracion 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    if lemas:
        corpus_lematizado.append(" ".join(lemas))

# 3. Implementar Bag of Words (BoW)
# Cuenta la frecuencia ignorando el orden [cite: 211, 212]
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)

# 4. Implementar TF-IDF
# Evalúa relevancia y reduce el peso de palabras muy comunes [cite: 243, 287]
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)

print(f"Vocabulario total: {len(tfidf_vectorizer.get_feature_names_out())} palabras únicas.")
print(f"Forma de la matriz TF-IDF: {X_tfidf.shape}")