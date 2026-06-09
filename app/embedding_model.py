import spacy


class EmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def get_embedding(self, word: str):
        token = self.nlp(word)[0]
        return token.vector.tolist()

    def get_similarity(self, word1: str, word2: str):
        token1 = self.nlp(word1)[0]
        token2 = self.nlp(word2)[0]
        return float(token1.similarity(token2))

    def has_vector(self, word: str):
        token = self.nlp(word)[0]
        return bool(token.has_vector)