import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. CARGA Y LIMPIEZA (INGESTA)
nlp = spacy.load("es_core_news_sm")
with open("libro.txt", "r", encoding="utf-8") as f:
    texto = f.read()

doc = nlp(texto)
sentences = []
for sent in doc.sents:
    tokens = [t.lemma_.lower() for t in sent if not t.is_stop and not t.is_punct and t.text.strip()]
    if len(tokens) > 1:
        sentences.append(tokens)

# 2. VECTORIZACIÓN DISTRIBUCIONAL (Word2Vec)
# Entrenamos para obtener vectores densos [cite: 373, 379]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 3. GENERACIÓN DE IMÁGENES (REDUCCIÓN PCA)
def guardar_grafica_3d(modelo, nombre_archivo, titulo):
    vocab = list(modelo.wv.index_to_key)
    vectores = modelo.wv[vocab]
    pca = PCA(n_components=3) # Reducimos a 3D para visualizar [cite: 168, 408]
    coords = pca.fit_transform(vectores)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='crimson')
    
    for i, word in enumerate(vocab[:20]): # Etiquetamos solo las primeras 20 para claridad
        ax.text(coords[i,0], coords[i,1], coords[i,2], word)
        
    plt.title(titulo)
    plt.savefig(nombre_archivo)
    print(f"Imagen {nombre_archivo} guardada.")

guardar_grafica_3d(model, "espacio_semantico_1.png", "Relaciones Semánticas - Vista A")
guardar_grafica_3d(model, "espacio_semantico_2.png", "Relaciones Semánticas - Vista B")