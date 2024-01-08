from sentence_transformers import SentenceTransformer

# Загрузка предобученной модели SBERT
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Предложение для векторизации
sentence = "This is an example sentence."

# Получение вектора предложения
sentence_vector = model.encode([sentence])[0]

# Вывод вектора
print("Вектор предложения:")
print(sentence_vector)
