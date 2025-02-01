import argparse
import pandas as pd
import numpy as np
from scipy.special import logsumexp

# Set the "divide" error mode to "ignore"
np.seterr(divide='ignore', invalid='ignore')


BLOCK_SIZE = 50
FIRST_STATE = 'State 1'
SECOND_STATE = 'State 2'


def forward_algorithm(X, emission, tau, p0):
    """
    the implementation for forward_algorithm
    :return: an array F is the results of forward_algorithm
    """
    num_states, seq_length = len(tau), len(X)
    F = np.full((num_states, seq_length), -np.inf)
    F[0][0] = np.log(p0) + np.log(emission.loc[FIRST_STATE, X[0]])
    F[1][0] = np.log(1 - p0) + np.log(emission.loc[SECOND_STATE, X[0]])

    for l in range(1, seq_length):
        for i in range(num_states):
            F[i][l] = logsumexp([F[i][l - 1] + np.log(tau[i][i]), F[abs(i - 1)][l - 1] + np.log(tau[abs(i - 1)][i])])\
                      + np.log(emission.loc[emission.index[i], X[l]])

    return np.exp(F)


def backward_algorithm(X, emission, tau, p0):
    """
    the implementation for backward algorithm
    :return: an array B is the results of backward algorithm
    """
    num_states, seq_length = len(tau), len(X)
    B = np.full((num_states, seq_length + 1), -np.inf)

    B[:, -1] = 0

    for l in range(seq_length - 1, 0, -1):
        for i in range(num_states):
            B[i][l] = logsumexp([
                np.log(tau[i][i]) + np.log(emission.loc[emission.index[i], X[l]]) + B[i][l + 1],
                np.log(tau[i][abs(i - 1)]) + np.log(emission.loc[emission.index[abs(i - 1)], X[l]]) + B[abs(i - 1)][l + 1]
            ])

    B[0][0] = np.log(p0) + np.log(emission.loc[FIRST_STATE, X[-1]]) + B[0][1]
    B[1][0] = np.log(1 - p0) + np.log(emission.loc[SECOND_STATE, X[0]]) + B[1][1]

    return np.exp(B)


def posterior_algorithm(X, emission, tau, p0):
    """
    gets F and B queries based on the forward and barebacked algorithm
    calculates the posterior and prints it
    """
    F = np.log(forward_algorithm(X, emission, tau, p0))
    B = np.log(backward_algorithm(X, emission, tau, p0))
    B = B[:, 1:]

    likelihood = logsumexp(F[:, -1])
    posterior_probabilities = np.exp((F + B) - likelihood)

    for i in range(len(posterior_probabilities[0])):
        print(f"{X[i]}\t{posterior_probabilities[0, i]:.3f}\t{posterior_probabilities[1, i]:.3f}")


def printing_results(best_align1, best_align2):
    """
    prints the results of the viterbi_algorithm
    """
    """ we print the results a line against the other line and the score"""
    for i in range(0, len(best_align1), BLOCK_SIZE):
        print(best_align1[i:i + BLOCK_SIZE])
        print(best_align2[i:i + BLOCK_SIZE])
        print()


def viterbi_algorithm(X, emission, tau, p0):
    """
    the implementation for viterbi algorithm
    """
    # initialize_matrices
    num_states, seq_length = len(tau), len(X)
    V = np.full((num_states, seq_length), -np.inf)
    backpointers = np.zeros((num_states, seq_length), dtype=int)  # Initialize back-pointers
    V[0][0] = np.log(p0) + np.log(emission.loc[FIRST_STATE, X[0]])
    V[1][0] = np.log(1 - p0) + np.log(emission.loc[SECOND_STATE, X[0]])

    # forward_pass
    for l in range(1, seq_length):
        for i in range(num_states):
            values = [
                V[i][l - 1] + np.log(tau[i][i]) + np.log(emission.loc[emission.index[i], X[l]]),
                V[abs(i - 1)][l - 1] + np.log(tau[abs(i - 1)][i]) + np.log(emission.loc[emission.index[i], X[l]])
            ]
            max_index = np.argmax(values)
            V[i][l] = values[max_index]
            backpointers[i][l] = max_index

    # backtrack_path
    best_path = [np.argmax(V[:, -1]) + 1]  # Initialize with the state of the last column
    for l in range(seq_length - 1, 0, -1):
        max_index_prev = backpointers[best_path[-1] - 1][l]
        best_path.append(3 - best_path[-1]) if max_index_prev != 0 else best_path.append(best_path[-1])
    best_path.reverse()
    best_path = ''.join(map(str, best_path))
    printing_results(best_path, X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq',
                        help='A sequence over the alphabet (e.g. ACTGGACTACGTCATGCA or 1621636142516152416616666166616)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emision.tsv)')
    parser.add_argument('p', help='probability to transition from state 1 to state 2 (e.g. 0.01)', type=float)
    parser.add_argument('q', help='probability to transition from state 2 to state 1 (e.g. 0.5)', type=float)
    parser.add_argument('p0', help='intial probability of entering state 1 (e.g. 0.9)', type=float)
    args = parser.parse_args()
    seq, p, q, p0 = str(args.seq), float(args.p), float(args.q), float(args.p0)

    # creating the emission
    emission = pd.read_csv(args.initial_emission, sep='\t')
    emission.index = [FIRST_STATE, SECOND_STATE]

    # creating the tau
    tau = np.array([[1 - p, p], [q, 1 - q]])

    if args.alg == 'viterbi':
        viterbi_algorithm(seq, emission, tau, p0)

    elif args.alg == 'forward':
        F = forward_algorithm(seq, emission, tau, p0)
        print(round(np.log2(np.sum(F[:, -1])), 2))

    elif args.alg == 'backward':
        B = backward_algorithm(seq, emission, tau, p0)
        print(round(np.log2(np.sum(B[:, 0])), 2))

    elif args.alg == 'posterior':
        posterior_algorithm(seq, emission, tau, p0)


if __name__ == '__main__':
    main()

# We can see the results are compatible with the posterior decoding.
# In the example you gave in the moodle we can see that viterbi starts with "state 2" and
# the posterior is higher in "state 2" then "state 1" but from the moment that it change from
# "state 2" to "state 1" in the viterbi also the posterior was higher in "state 1" then from "state 2"
# The compatibility between Posterior Decoding and Viterbi Decoding is ensured mathematically.

# The alignment of the most probable state in the Viterbi path,argmaxP(X,Q) with the state having
# the highest posterior probability at each time step,argmax P(qt∣X).
# This alignment signifies that the sequence of hidden states found by
# Viterbi maximizes the joint probability consistently with the posterior
# probabilities assigned to each state by Posterior Decoding.