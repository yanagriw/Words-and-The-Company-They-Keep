from collections import defaultdict, Counter
import math
import numpy as np


# Initializes class mapping based on word frequency, excluding words with frequency below a threshold.
def init_class_map(words):
    freq_words = Counter(words)  # Counts the frequency of each word in the list.
    return {word: word for word in words}, {
        word: word
        for word in words
        if freq_words[word] >= 10
        # UNCOMMENT FOR THE 3RD TASK
        # word: word for word in words if freq_words[word] >= 5
    }


# Computes counts for bigrams (c_bigram), left words (c_left), and right words (c_right) from word pairs.
def compute_counts(word_pairs, class_map):
    c_bigram = defaultdict(int)
    c_left = defaultdict(int)
    c_right = defaultdict(int)

    for word1, word2 in word_pairs:
        class1 = class_map[word1]  # Maps word to its class.
        class2 = class_map[word2]
        c_bigram[class1, class2] += 1
        c_left[class1] += 1
        c_right[class2] += 1

    return c_bigram, c_left, c_right


# Computes pointwise mutual information (PMI) for all bigrams.
def compute_all_q(c_bigram, c_left, c_right):
    q = defaultdict(float)
    N = sum(c_bigram.values())  # Total number of bigrams.
    for (l, r), count in c_bigram.items():
        q[l, r] = compute_q(N, count, c_left[l], c_right[r])
    return q


# Calculates PMI for a given bigram.
def compute_q(N, c_bigram, c_left, c_right):
    return c_bigram / N * math.log(N * (c_bigram / (c_left * c_right)), 2)

# Initializes scores for each word based on its PMI values.
def init_s(q):
    s = defaultdict(float)
    for (l, r), q_lr in q.items():
        s[l] += q_lr
        s[r] += q_lr
    for l in s:
        s[l] -= q.get((l, l), 0) # Adjusts score by subtracting self-pair PMI if present.
    return s

# Identifies the pair of words with the minimum mutual information loss.
def init_L(s, q, ck_bigram, ck_left, ck_right, filtered_words):

    L = defaultdict(float)
    N = sum(ck_bigram.values())
    e = 1e-10 # Small epsilon to avoid division by zero.

    min_loss = float("inf")
    best_pair = None

    # Iterates over all possible pairs of filtered words to calculate their MI loss.
    for i in range(len(filtered_words)):
        for j in range(i + 1, len(filtered_words)):

            a = filtered_words[i]
            b = filtered_words[j]

            L[a, b] = s[a] + s[b] - q[(a, b)] - q.get((b, a), e)

            L[a, b] -= compute_q(
                N,
                ck_bigram.get((a, b), e)
                + ck_bigram.get((b, a), e)
                + ck_bigram.get((a, a), e)
                + ck_bigram.get((b, b), e),
                ck_left.get(a, e) + ck_left.get(b, e),
                ck_right.get(a, e) + ck_right.get(b, e),
            )

            for l, l_count in ck_left.items():
                if l != a and l != b:
                    L[a, b] -= compute_q(
                        N,
                        ck_bigram.get((l, a), e) + ck_bigram.get((l, b), e),
                        l_count,
                        ck_right.get(a, e) + ck_right.get(b, e),
                    )

            for r, r_count in ck_right.items():
                if r != a and r != b:
                    L[a, b] -= compute_q(
                        N,
                        ck_bigram.get((a, r), e) + ck_bigram.get((b, r), e),
                        ck_left.get(a, e) + ck_left.get(b, e),
                        r_count,
                    )

            if L[a, b] < min_loss:
                min_loss = L[a, b]
                best_pair = (a, b)

    return L, best_pair


def average_mut_inf(q):
    return sum(q.values())


def merge_classes(class1, class2, class_map, filtered_class_map):
    for w, c in filtered_class_map.items():
        if c == class2:
            filtered_class_map[w] = class1
            class_map[w] = class1
    return class_map, filtered_class_map

# Greedy clustering algorithm to find best word association pairs based on mutual information.
def greedy_clustering(words, filename, desired_classes=1):
    with open(filename, "w") as file:
        e = 1e-16
        word_pairs = list(zip(words, words[1:]))
        class_map, class_map_filtered = init_class_map(words)
        filtered_words = list(class_map_filtered.values())

        ck_bigram, ck_left, ck_right = compute_counts(word_pairs, class_map)
        N = sum(ck_bigram.values())
        qk = compute_all_q(ck_bigram, ck_left, ck_right)
        sk = init_s(qk)
        Lk, best_pair = init_L(
            sk, qk, ck_bigram, ck_left, ck_right, list(class_map_filtered.values())
        )
        print(best_pair)
        file.write(f"{best_pair}\n")

        a, b = best_pair
        class_map, class_map_filtered = merge_classes(
            a, b, class_map, class_map_filtered
        )
        while len(set(class_map_filtered.values())) > desired_classes:
            ck_new_bigram, ck_new_left, ck_new_right = compute_counts(
                word_pairs, class_map
            )
            qk_new = compute_all_q(ck_new_bigram, ck_new_left, ck_new_right)

            sk_new = defaultdict(float)
            for i in words:
                i = class_map[i]
                if i != a:
                    sk_new[i] = (
                        sk[i]
                        - qk.get((i, a), e)
                        - qk.get((a, i), e)
                        - qk.get((i, b), e)
                        - qk.get((b, i), e)
                        + qk_new.get((a, i), e)
                        + qk_new.get((i, a), e)
                    )
                else:
                    sk_new[i] = init_s(qk_new)[i]

            min_loss = float("inf")
            best_pair = None

            Lk_new = defaultdict(float)
            for i in range(len(filtered_words)):
                for j in range(i + 1, len(filtered_words)):
                    i_word = class_map_filtered[filtered_words[i]]
                    j_word = class_map_filtered[filtered_words[j]]
                    if i_word != j_word:
                        if i_word != a and j_word != a:
                            Lk_new[i_word, j_word] = (
                                Lk[i_word, j_word]
                                - sk[i_word]
                                + sk_new[i_word]
                                - sk[j_word]
                                + sk_new[j_word]
                            )
                            Lk_new[i_word, j_word] += compute_q(
                                N,
                                ck_bigram.get((i_word, a), e)
                                + ck_bigram.get((j_word, a), e),
                                ck_left.get(i_word, e) + ck_left.get(j_word, e),
                                ck_right.get(a, e),
                            )
                            Lk_new[i_word, j_word] += compute_q(
                                N,
                                ck_bigram.get((a, i_word), e)
                                + ck_bigram.get((a, j_word), e),
                                ck_left.get(a, e),
                                ck_right.get(i_word, e) + ck_right.get(j_word, e),
                            )
                            Lk_new[i_word, j_word] += compute_q(
                                N,
                                ck_bigram.get((i_word, b), e)
                                + ck_bigram.get((j_word, b), e),
                                ck_left.get(i_word, e) + ck_left.get(j_word, e),
                                ck_right.get(b, e),
                            )
                            Lk_new[i_word, j_word] += compute_q(
                                N,
                                ck_bigram.get((b, i_word), e)
                                + ck_bigram.get((b, j_word), e),
                                ck_left.get(b, e),
                                ck_right.get(i_word, e) + ck_right.get(j_word, e),
                            )
                            Lk_new[i_word, j_word] -= compute_q(
                                N,
                                ck_new_bigram.get((i_word, a), e)
                                + ck_new_bigram.get((j_word, a), e),
                                ck_new_left.get(i_word, e) + ck_new_left.get(j_word, e),
                                ck_new_right.get(a, e),
                            )
                            Lk_new[i_word, j_word] -= compute_q(
                                N,
                                ck_new_bigram.get((a, i_word), e)
                                + ck_new_bigram.get((a, j_word), e),
                                ck_new_left.get(a, e),
                                ck_new_right.get(i_word, e)
                                + ck_new_right.get(j_word, e),
                            )

                        else:
                            Lk_new[i_word, j_word] = (
                                sk_new[i_word]
                                + sk_new[j_word]
                                - qk_new[(i_word, j_word)]
                                - qk_new.get((j_word, i_word), e)
                            )

                            Lk_new[i_word, j_word] -= compute_q(
                                N,
                                ck_new_bigram.get((i_word, j_word), e)
                                + ck_new_bigram.get((j_word, i_word), e)
                                + ck_new_bigram.get((i_word, i_word), e)
                                + ck_new_bigram.get((j_word, j_word), e),
                                ck_new_left.get(i_word, e) + ck_new_left.get(j_word, e),
                                ck_new_right.get(i_word, e)
                                + ck_new_right.get(j_word, e),
                            )

                            for l, l_count in ck_new_left.items():
                                if l != i_word and l != j_word:
                                    Lk_new[i_word, j_word] -= compute_q(
                                        N,
                                        ck_new_bigram.get((l, i_word), e)
                                        + ck_new_bigram.get((l, j_word), e),
                                        l_count,
                                        ck_new_right.get(i_word, e)
                                        + ck_new_right.get(j_word, e),
                                    )

                            for r, r_count in ck_new_right.items():
                                if r != i_word and r != j_word:
                                    Lk_new[i_word, j_word] -= compute_q(
                                        N,
                                        ck_new_bigram.get((i_word, r), e)
                                        + ck_new_bigram.get((j_word, r), e),
                                        ck_new_left.get(i_word, e)
                                        + ck_new_left.get(j_word, e),
                                        r_count,
                                    )

                        if Lk_new[i_word, j_word] < min_loss:
                            min_loss = Lk_new[i_word, j_word]
                            best_pair = (i_word, j_word)

            print(best_pair)
            file.write(f"{best_pair}\n")
            a, b = best_pair
            class_map, class_map_filtered = merge_classes(
                a, b, class_map, class_map_filtered
            )
            ck_bigram, ck_left, ck_right = ck_new_bigram, ck_new_left, ck_new_right
            qk = qk_new
            sk = sk_new
            Lk = Lk_new

            if len(set(class_map_filtered.values())) == 15:
                class_map_with_15_classes = class_map_filtered.copy()

    return class_map_with_15_classes


def print_classes(class_map, filename):
    class_words = defaultdict(list)
    # Invert the dictionary to group words by class
    for word, class_ in class_map.items():
        class_words[class_].append(word)

    # Write words of the same class into a file
    with open(filename, "w") as file:
        for class_, words in class_words.items():
            file.write(f"Class: {class_}\n")
            file.write("\t")
            for word in words:
                file.write(f"{word}\t")
            file.write("\n\n")


def main():
    with open("TEXTCZ1.ptg", "r", encoding="iso-8859-2") as file:
        text_cz = [line.strip().split("/")[0] for line in file.readlines()][:8000]
    with open("TEXTEN1.ptg", "r", encoding="iso-8859-2") as file:
        text_en = [line.strip().split("/")[0] for line in file.readlines()][:8000]

    print("Clustering for en text...")
    class_map_en = greedy_clustering(text_en, "history_en.txt", 1)
    print("Clustering for cz text...")
    class_map_cz = greedy_clustering(text_cz, "history_cz.txt", 15)

    print_classes(class_map_en, "words_by_class_en.txt")
    print_classes(class_map_cz, "words_by_class_cz.txt")

    # UNCOMMENT FOR THE 3RD TASK

    # with open("TEXTCZ1.ptg", "r", encoding="iso-8859-2") as file:
    #     text_cz = [line.strip().split("/")[1] for line in file.readlines()]
    # with open("TEXTEN1.ptg", "r", encoding="iso-8859-2") as file:
    #     text_en = [line.strip().split("/")[1] for line in file.readlines()]

    # print("Clustering for en text...")
    # class_map_en = greedy_clustering(text_en, "tags_history_en.txt", 1)
    # print("Clustering for cz text...")
    # class_map_cz = greedy_clustering(text_cz, "tags_history_cz.txt", 15)

    # print_classes(class_map_en, "tags_by_class_en.txt")
    # print_classes(class_map_cz, "tags_by_class_cz.txt")


if __name__ == "__main__":
    main()
