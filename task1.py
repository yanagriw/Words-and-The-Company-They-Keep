from collections import Counter
import math
import pandas as pd

def compute_pmi(pair_freq, word_freq, total_pairs, total_words):
    pmi_distant = {}
    for pair, freq in pair_freq.items():
        p_x_y = freq / total_pairs
        p_x = word_freq[pair[0]] / total_words
        p_y = word_freq[pair[1]] / total_words
        pmi_distant[pair] = math.log(p_x_y / (p_x * p_y), 2)

    # Sort and display the top 20 pairs
    top_20_distant = sorted(pmi_distant.items(), key=lambda x: x[1], reverse=True)[:20]
    print(pd.DataFrame(top_20_distant, columns=['Pair', 'PMI']))

def consecutive_words(words):
    # Compute frequencies
    word_freq = Counter(words)
    total_words = len(words)
    total_pairs = total_words - 1

    # Compute pair frequencies for consecutive words and Filter out words with frequency < 10
    pair_freq_consecutive = Counter()
    for i in range(len(words) - 1):
        if word_freq[words[i]] >= 10 and word_freq[words[i+1]] >= 10:
            pair = (words[i], words[i+1])
            pair_freq_consecutive[pair] += 1

    compute_pmi(pair_freq_consecutive, word_freq, total_pairs, total_words)

def distant_words(words):
    # Compute frequencies
    word_freq = Counter(words)
    total_words = len(words)
    total_pairs = 0

    # Compute pair frequencies for distant words (at least 1 word apart, up to 50 words apart)
    pair_freq_distant = Counter()

    for i in range(len(words)):
        for j in range(max(0, i-50), i - 1):
            total_pairs += 1

            if word_freq[words[i]] >= 10 and word_freq[words[j]] >= 10:
                pair = (words[i], words[j])
                pair_freq_distant[pair] += 1

        for j in range (i + 2, min(len(words), i+51)):
            total_pairs += 1
            
            if word_freq[words[i]] >= 10 and word_freq[words[j]] >= 10:
                pair = (words[i], words[j])
                pair_freq_distant[pair] += 1


    compute_pmi(pair_freq_distant, word_freq, total_pairs, total_words)

def main():
    with open("TEXTCZ1.txt", 'r', encoding="iso-8859-2") as file:
        text_cz = file.read().split('\n')
    with open("TEXTEN1.txt", 'r', encoding="iso-8859-2") as file:
        text_en = file.read().split('\n')

    # Tokenize each line into words and characters
    tokens_cz = [line for line in text_cz if line != '']
    tokens_en = [line for line in text_en if line != '']
    print("English Text:")
    consecutive_words(tokens_en)
    distant_words(tokens_en)

    print("Czech Text:")
    consecutive_words(tokens_cz)
    distant_words(tokens_cz)

if __name__ == "__main__":
    main()