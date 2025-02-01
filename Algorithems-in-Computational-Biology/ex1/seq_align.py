import argparse
import numpy as np
import pandas as pd
from itertools import groupby
import psutil
import time
import os

BLOCK_SIZE = 50
DASH = '-'


def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    In Ex1 you may assume the fasta files contain only one sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'seq_a', help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument(
        '--align_type', help='Alignment type (e.g. local)', required=True)
    parser.add_argument(
        '--score', help='Score matrix in.tsv format (default is score_matrix.tsv) ', default='score_matrix.tsv')
    command_args = parser.parse_args()

    args = parser.parse_args()

    seq_a_path = args.seq_a
    seq_b_path = args.seq_b
    align_type = args.align_type
    score_matrix = args.score

    seq_a = fastaread(seq_a_path).__next__()[1]
    seq_b = fastaread(seq_b_path).__next__()[1]

    df_score = pd.read_csv(args.score, sep='\t', index_col=0)

    df = init_score_matrix(seq_a, seq_b, df_score, command_args.align_type)

    if command_args.align_type == 'global':
        trace_matrix = init_trace_matrix(len(seq_a), len(seq_b))

        df, trace_matrix = fill_score_matrix(df, df_score, trace_matrix)
        score = df.iloc[-1, -1]

        best_align1, best_align2 = finding_results_global_local(trace_matrix, seq_a, seq_b, len(seq_a), len(seq_b))
        printing_results(best_align1, best_align2, score, align_type)

    elif command_args.align_type == 'local':
        trace_matrix = init_trace_matrix_local(len(seq_a), len(seq_b))

        df, trace_matrix = fill_score_matrix(df, df_score, trace_matrix, True)

        best_row, best_col = divmod(df.values.argmax(), df.shape[1])

        score = df.iloc[best_row, best_col]

        best_align1, best_align2 = finding_results_global_local(trace_matrix, seq_a, seq_b, best_row, best_col)
        printing_results(best_align1, best_align2, score, align_type)

    elif command_args.align_type == 'overlap':
        trace_matrix = init_trace_matrix(len(seq_a), len(seq_b))

        df, trace_matrix = fill_score_matrix(df, df_score, trace_matrix)
        best_col1, best_row1 = df.shape[1] - 1, np.argmax(df.iloc[:, -1].values)
        score_rows = df.iloc[best_row1, best_col1]

        best_row2, best_col2 = df.shape[0] - 1, np.argmax(df.iloc[df.shape[0] - 1].values)
        score_col = df.iloc[best_row2, best_col2]
        if score_rows > score_col:
            best_align1, best_align2 = finding_results_overlap(trace_matrix, seq_a, seq_b, best_row1, best_col1)
            printing_results(best_align1, best_align2, score_rows, align_type)
        else:
            best_align1, best_align2 = finding_results_overlap(trace_matrix, seq_a, seq_b, best_row2, best_col2, False)
            printing_results(best_align1, best_align2, score_col, align_type)

    elif command_args.align_type == 'global_lin':
        return


def calculate_score_two_letters(let1, let2, score_matrix):
    """return the matching cell in the score_matrix"""
    return score_matrix.loc[let1, let2]


def init_score_matrix(seq1, seq2, score_matrix, align_type):
    """create new score data frame, and init the first row and the first col."""

    seq1 = DASH + seq1
    seq2 = DASH + seq2
    df = pd.DataFrame(0, index=list(seq1), columns=list(seq2))
    if align_type == 'global':
        for i in range(1, len(seq1)):
            df.iloc[i, 0] = df.iloc[i - 1, 0] + \
                            calculate_score_two_letters(df.index[i], DASH, score_matrix)
        for j in range(1, len(seq2)):
            df.iloc[0, j] = df.iloc[0, j - 1] + \
                            calculate_score_two_letters(DASH, df.columns[j], score_matrix)
    return df


def init_score_matrix_overlap(seq1, seq2, score_matrix):
    """create new score data frame, and init the first row and the first col."""

    seq1 = DASH + seq1
    seq2 = DASH + seq2
    df1 = pd.DataFrame(0, index=list(seq1), columns=list(seq2))
    df2 = pd.DataFrame(0, index=list(seq1), columns=list(seq2))
    for i in range(1, len(seq1)):
        df1.iloc[i, 0] = df1.iloc[i - 1, 0] + \
                        calculate_score_two_letters(df1.index[i], DASH, score_matrix)
    for j in range(1, len(seq2)):
        df2.iloc[0, j] = df2.iloc[0, j - 1] + \
                        calculate_score_two_letters(DASH, df2.columns[j], score_matrix)

    return df1, df2


def init_trace_matrix(len_seq1, len_seq2):
    """
    init new matrix that represtent the trace matrix.
    """

    result = np.full((len_seq1 + 1, len_seq2 + 1), 7)
    result[:, 0] = 1  # Set the first column to 0
    result[0, 1:] = 0
    return result


def init_trace_matrix_local(len_seq1, len_seq2):
    """
    init new matrix that represtent the trace matrix.
    """

    result = np.full((len_seq1 + 1, len_seq2 + 1), 7)
    result[:, 0] = 3  # Set the first column to 0
    result[0, 1:] = 3
    return result


def finding_results_global_local(trace_matrix, seq_a, seq_b, row_index, col_index):
    """finds the base results for local and global"""

    best_align1 = best_align2 = ""
    i, j = row_index, col_index
    seq_a, seq_b = "-" + seq_a, "-" + seq_b

    while True:
        if i <= 0 and j <= 0:
            break
        direction = trace_matrix[i][j]
        if direction == 2:
            best_align1 = seq_a[i] + best_align1
            best_align2 = seq_b[j] + best_align2
            i -= 1
            j -= 1
        elif direction == 0:
            best_align1 = DASH + best_align1
            best_align2 = seq_b[j] + best_align2
            j -= 1

        elif direction == 1:
            best_align1 = seq_a[i] + best_align1
            best_align2 = DASH + best_align2
            i -= 1
        else:
            break
    return best_align1, best_align2


def finding_results_overlap(trace_matrix, seq_a, seq_b, row_index, col_index, align_by_row=True):
    """finds the base results for overlap"""

    best_align1 = best_align2 = ""
    i, j = row_index, col_index
    seq_a, seq_b = "-" + seq_a, "-" + seq_b
    k = trace_matrix.shape[0] - 1 if align_by_row else trace_matrix.shape[1] - 1

    while (k > i) if align_by_row else (k > j):
        best_align1 = seq_a[k] + best_align1 if align_by_row else DASH + best_align1
        best_align2 = DASH + best_align2 if align_by_row else seq_b[k] + best_align2
        k -= 1

    while (i > 0) if align_by_row else (j > 0):
        direction = trace_matrix[i][j]
        if direction == 2:
            best_align1 = seq_a[i] + best_align1
            best_align2 = seq_b[j] + best_align2
            i -= 1
            j -= 1
        elif direction == 1:
            best_align1 = seq_a[i] + best_align1
            best_align2 = DASH + best_align2
            i -= 1
        elif direction == 0:
            best_align1 = DASH + best_align1
            best_align2 = seq_b[j] + best_align2
            j -= 1

    while (j > 0) if align_by_row else (i > 0):
        if align_by_row:
            best_align2 = seq_b[j] + best_align2
            best_align1 = DASH + best_align1
            j -= 1
        else:
            best_align1 = seq_a[i] + best_align1
            best_align2 = DASH + best_align2
            i -= 1

    return best_align1, best_align2


def fill_score_matrix(df, score_matrix, trace_matrix, local=False):
    """fill the matching score to each cell in the matrix"""

    for i in range(1, df.shape[0]):
        for j in range(1, df.shape[1]):
            left_row_score = df.iloc[i, j-1] + \
                calculate_score_two_letters(DASH, df.columns[j], score_matrix)
            top_col_score = df.iloc[i-1, j] + \
                calculate_score_two_letters(df.index[i], DASH, score_matrix)
            diagonal_score = df.iloc[i-1, j-1] + calculate_score_two_letters(
                df.index[i], df.columns[j], score_matrix)

            if local:
                maxes = [left_row_score, top_col_score, diagonal_score, 0]
            else:
                maxes = [left_row_score, top_col_score, diagonal_score]

            max_score = max(maxes)
            best_index = maxes.index(max_score)

            df.iloc[i, j] = max_score
            trace_matrix[i, j] = best_index

    return df, trace_matrix


def find_max_index(df):
    """find the cell with the highest score in the last col and the last row"""
    max_index_last_row = df.iloc[-1, :].idxmax()
    max_index_last_col = df.iloc[:, -1].idxmax()
    return df.iloc[-1, max_index_last_row], max_index_last_col


def printing_results(best_align1, best_align2, score, align_type):
    """ we print the results a line against the other line and the score"""
    for i in range(0, len(best_align1), BLOCK_SIZE):
        print(best_align1[i:i + BLOCK_SIZE])
        print(best_align2[i:i + BLOCK_SIZE])
        print()

    print(align_type + ":" + str(score))


if __name__ == '__main__':
    main()