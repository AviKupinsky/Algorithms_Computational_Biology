import sys
import numpy as np
from Bio import Phylo
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

# Set the "divide" error mode to "ignore"
np.seterr(divide='ignore', invalid='ignore')

# Define the Jukes-Cantor rate matrix
Q = np.array([[-3, 1, 1, 1],
              [1, -3, 1, 1],
              [1, 1, -3, 1],
              [1, 1, 1, -3]])

# Nucleotide to index mapping
nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': -1}


# Parse sequences from the given string format
def parse_sequences(seq_str):
    """
    Parses a string containing multiple sequences in a simple format
    :param seq_str: A string containing the sequences to be parsed
    :return: A dictionary where each key is a sequence identifier (from the input string) and each value is
            the corresponding sequence as a string.
    """
    lines = seq_str.strip().split('\n')
    seq_data = lines[1:]  # Skip the first line with counts
    sequences_dicy = {}
    for line in seq_data:
        parts = line.split()
        sequences_dicy[parts[0]] = parts[1]
    return sequences_dicy


def calculate_likelihood(alpha, window_seqs, tree):
    """
    Calculates the log-likelihood of observing windowed sequences given a phylogenetic tree and alpha.

    Parameters:
    - alpha (float): The conservation coefficient to adjust branch lengths in the tree.
    - window_seqs (dict): Dictionary of sequence IDs and their corresponding windowed sequences.
    - tree (Bio.Phylo.BaseTree.Tree): The phylogenetic tree relating the sequences.

    Returns:
    - float: The summed log-likelihood for observing the given sequences across all windows.

    This function assumes sequences are related as per the phylogenetic tree, and uses
    the Jukes-Cantor model for nucleotide substitution probabilities.
    """
    log_likelihood = 0.0
    for clade in tree.find_clades():
        if clade is not tree.root:
            parent = tree.get_path(clade)[-2] if len(tree.get_path(clade)) >= 2 else tree.root
            adjusted_length = alpha * clade.branch_length
            P = expm(Q * adjusted_length)

            for position, orig_nuc in enumerate(window_seqs[parent.name]):
                seq_nuc = window_seqs[clade.name][position]
                if seq_nuc not in ['A', 'C', 'G', 'T'] or orig_nuc not in ['A', 'C', 'G', 'T']:
                    continue
                prob = P[nucleotide_to_index[orig_nuc], nucleotide_to_index[seq_nuc]]
                log_likelihood += np.log(prob)


    return log_likelihood


def optimize_alpha(sequences, tree, window_size=11):
    """
    Optimizes alpha for each window across the sequences to maximize log-likelihood.

    Parameters:
    - sequences (dict): Sequence IDs mapped to their respective sequences.
    - tree (Bio.Phylo.BaseTree.Tree): Phylogenetic tree of the sequences.
    - window_size (int): Size of the window for sequence analysis (default: 11).

    Returns:
    - list: Optimal alpha values for each window to maximize the log-likelihood.

    This function employs a scalar minimization technique to find the alpha that
    yields the highest log-likelihood for each window, given the phylogenetic tree
    and the windowed sequences.
    """

    def objective(alpha, window_seqs, tree):
        # Negate log_likelihood because minimize_scalar seeks to minimize the function
        return -calculate_likelihood(alpha, window_seqs, tree)

    seq_length = len(next(iter(sequences.values())))
    optimized_alphas = []

    for start in range(seq_length - window_size + 1):
        window_seqs = {name: seq[start:start + window_size] for name, seq in sequences.items()}

        # Use minimize_scalar with bounded method for finding optimal alpha
        res = minimize_scalar(objective, bounds=(0, 0.1), args=(window_seqs, tree), method='bounded')

        if res.success:
            optimized_alphas.append(res.x)
        else:
            # Handle failure case; could log an error or use a default value
            optimized_alphas.append(None)

    return optimized_alphas


if __name__ == '__main__':
    # Load the tree from the Newick string
    tree = Phylo.read(sys.argv[1], "newick")
    with open(sys.argv[2], 'r') as file:
        file_contents = file.read()

    sequences = parse_sequences(file_contents)
    # Perform the optimization to find alpha for each window
    optimized_alphas = optimize_alpha(sequences, tree)

    # Print the optimized alphas with 3 decimal places
    for alpha in optimized_alphas:
        formatted_alpha = f"{alpha:.3f}".rstrip('0').rstrip('.')
        if formatted_alpha == "0":
            print(0.0)
        else:
            print(formatted_alpha)
