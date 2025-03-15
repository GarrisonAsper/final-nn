# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    import random

    #separate positive and negative sequences
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label == 1]  # Explicit check
    neg_seqs = [seq for seq, label in zip(seqs, labels) if label == 0]


    #determine the minority class size
    min_class_size = min(len(pos_seqs), len(neg_seqs))

    #sample with replacement to balance classes
    sampled_pos = random.choices(pos_seqs, k=min_class_size)
    sampled_neg = random.choices(neg_seqs, k=min_class_size)

    #combine sampled sequences and shuffle
    sampled_seqs = sampled_pos + sampled_neg
    sampled_labels = [True] * min_class_size + [False] * min_class_size

    #shuffle dataset
    combined = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(combined)
    sampled_seqs, sampled_labels = zip(*combined)

    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    #mapping nucleotides to one-hot vectors
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}

    #encode sequences
    encoded_seqs = []
    for seq in seq_arr:
        one_hot_seq = [mapping[nt] for nt in seq]  
        flattened_seq = np.concatenate(one_hot_seq)  
        encoded_seqs.append(flattened_seq)

    return np.array(encoded_seqs)