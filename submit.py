import pandas as pd
import itertools

# Function to generate bigrams
def generate_bigrams(word):
    return sorted(list(set([word[i:i+2] for i in range(len(word)-1)])))

# Custom decision tree node
class TreeNode:
    def __init__(self, bigram=None, words=None, left=None, right=None):
        self.bigram = bigram
        self.words = words
        self.left = left
        self.right = right

# Function to build the custom decision tree
def build_tree(words, bigrams, required_bigrams, depth=0, max_depth=None):
    if words.empty or not bigrams or (max_depth and depth >= max_depth):
        return TreeNode(words=words)

    bigram = bigrams[0]
    left_words = words[words[bigram] == 1]
    right_words = words[words[bigram] == 0]

    if left_words.empty and right_words.empty:
        return TreeNode(words=words)

    required_bigrams.add(bigram)
    left_node = build_tree(left_words, bigrams[1:], required_bigrams, depth + 1, max_depth)
    right_node = build_tree(right_words, bigrams[1:], required_bigrams, depth + 1, max_depth)

    return TreeNode(bigram=bigram, left=left_node, right=right_node)

# Function to predict using the custom decision tree
def predict(tree, feature_vector):
    if tree.words is not None:
        result = (tree.words['Word'].tolist(), [1.0] * len(tree.words))
        return result

    if feature_vector.get(tree.bigram, 0) == 1:
        result = predict(tree.left, feature_vector)
    else:
        result = predict(tree.right, feature_vector)
    return result

# Function to create a feature vector for input bigrams
def create_feature_vector(input_bigrams, possible_bigrams):
    return {bigram: 1 if bigram in input_bigrams else 0 for bigram in possible_bigrams}

# Function to filter words containing all input bigrams
def filter_words_containing_bigrams(words, bigrams):
    return [word for word in words if all(bigram in generate_bigrams(word) for bigram in bigrams)]

def my_fit(words):
    df = pd.DataFrame({'Word': words})
    df['Bigrams'] = df['Word'].apply(generate_bigrams)
    df['test_bigrams'] = df['Bigrams'].apply(lambda bigrams: bigrams[:5])

    all_bigrams = set(itertools.chain(*df['Bigrams']))
    possible_bigrams = sorted(list(all_bigrams))

    for bigram in possible_bigrams:
        df[bigram] = df['Bigrams'].apply(lambda x: 1 if bigram in x else 0)

    required_bigrams = set()
    tree = build_tree(df, possible_bigrams, required_bigrams, max_depth=65)
    return tree, possible_bigrams

def my_predict(model, bg):
    tree, possible_bigrams = model
    input_feature_vector = create_feature_vector(bg, possible_bigrams)
    predicted_words, probabilities = predict(tree, input_feature_vector)
    filtered_predicted_words = filter_words_containing_bigrams(predicted_words, bg)

    word_prob_pairs = list(zip(filtered_predicted_words, probabilities[:len(filtered_predicted_words)]))
    sorted_word_prob_pairs = sorted(word_prob_pairs, key=lambda x: x[1], reverse=True)

    top_5_predictions = sorted_word_prob_pairs[:1]
    top_5_words = [pair[0] for pair in top_5_predictions]

    return top_5_words
