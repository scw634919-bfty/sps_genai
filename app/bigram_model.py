import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        self.frequency_threshold = frequency_threshold
        self.vocab = []
        self.bigram_probs = defaultdict(dict)
        self.train(corpus)

    def simple_tokenizer(self, text):
        tokens = re.findall(r"\b\w+\b", text.lower())

        if not self.frequency_threshold:
            return tokens

        word_counts = Counter(tokens)
        filtered_tokens = [
            token for token in tokens
            if word_counts[token] >= self.frequency_threshold
        ]

        return filtered_tokens

    def train(self, corpus):
        if isinstance(corpus, list):
            text = " ".join(corpus)
        else:
            text = corpus

        words = self.simple_tokenizer(text)

        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        self.vocab = list(unigram_counts.keys())

        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = count / unigram_counts[word1]

    def generate_text(self, start_word, num_words=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)

            if not next_words:
                break

            next_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values())
            )[0]

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)