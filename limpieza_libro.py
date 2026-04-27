import spacy

# 1. CARGA DEL MODELO
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# 2. INGESTA DE DATOS
with open("libro.txt", "r", encoding="utf-8") as f:
    texto_libro = f.read()

# 3. PROCESAMIENTO (Tokenización)
doc = nlp(texto_libro)

# 4. NORMALIZACIÓN Y LEMATIZACIÓN
tokens_normalizados = []

for token in doc:
    # FILTRADO DE RUIDO: Eliminamos Stop Words y puntuación [cite: 317]
    if not token.is_stop and not token.is_punct and token.text.strip():
        # LEMATIZACIÓN: Reducimos a la forma base [cite: 356]
        lema = token.lemma_.lower()
        tokens_normalizados.append(lema)

# 5. SALIDA DE DATOS
with open("libro_limpio.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(tokens_normalizados))

# LÍNEA CORREGIDA:
print(f"Reducción de dimensionalidad: de {len(doc)} a {len(tokens_normalizados)} tokens.")