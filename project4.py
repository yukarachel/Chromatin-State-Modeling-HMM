import numpy as np
import zipfile
from pathlib import Path
from collections import Counter

def initialize(path):
    symbol_map = {"x":0, "y":1, "z":2, "n":3}

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() in symbol_map]

    marks = np.array([symbol_map[l] for l in lines])
    counts = Counter(marks)
    total = len(marks)
    for sym, idx in symbol_map.items():
        print(f"{sym}: {counts[idx]/total:.4f}")

    return marks

def forward_backward(emission_num, emissions, states, transition_matrix, emission_matrix, initial_prob=0.5):
    '''takes in an emission string, list of emissions, list of states, transition matrix, and emission matrix. 
    Returns gamma, probs of each state at each position'''
    L = len(emission_num)
    S = len(states)
    forward_vars = np.zeros((L, S))
    backward_vars = np.zeros((L, S))
    scale_factors = np.zeros(L)
    
    # forward-----------------
    forward_vars[0, :] = initial_prob * emission_matrix[:, emission_num[0]]
    scale_factors[0] = forward_vars[0, :].sum()
    forward_vars[0, :] /= scale_factors[0] # scale to prevent underflow

    for i in range(1, L):
        for j in range(S): # current states
            forward_vars[i, j] = emission_matrix[j, emission_num[i]] * np.sum(forward_vars[i - 1, :] * transition_matrix[:, j])
        scale_factors[i] = forward_vars[i, :].sum()
        forward_vars[i, :] /= scale_factors[i]

    # backward-----------------
    backward_vars[L - 1, :] = 1 / scale_factors[L - 1]
    for i in range(L - 2, -1, -1):
        for j in range(S): # current states
            backward_vars[i, j] = np.sum(emission_matrix[:, emission_num[i + 1]] * backward_vars[i + 1, :] * transition_matrix[j, :])
        backward_vars[i, :] /= scale_factors[i] 

    # compute gamma-----------------
    gamma = forward_vars * backward_vars
    gamma /= gamma.sum(axis=1, keepdims=True) # normalize to get probabilities
    log_likelihood = np.sum(np.log(scale_factors)) # overall prob of gamma

    # estimate transition matrix 
    xi = np.zeros((L - 1, S, S))
    for i in range(L - 1):
        for j in range(S):
            for k in range(S):
                xi[i, j, k] = (forward_vars[i, j]
                * transition_matrix[j, k]
                * emission_matrix[k, emission_num[i + 1]]
                * backward_vars[i + 1, k])
                
        xi[i] /= xi[i].sum() # normalize to get probabilities by position

    # new transition matrix
    transition_matrix = xi.sum(axis=0)  # sum over positions
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)  # row-normalize

    # new emission matrix
    new_emission = np.zeros_like(emission_matrix)
    for e in range(len(emissions)):
        new_emission[:, e] = gamma[emission_num == e].sum(axis=0)  # sum over positions where emission matches
    emission_matrix = new_emission / new_emission.sum(axis=1, keepdims=True)  # row-normalize

    return transition_matrix, emission_matrix, gamma, log_likelihood

def baum_welch(emission_num, emissions, states, transition_matrix, emission_matrix, iterations, test, initial_prob=0.5):
    print("TESTING", test)

    prev_log_likelihood = None
    if test == "gene" and Path("g_final_emission_matrix.npy").exists() and Path("g_final_transition_matrix.npy").exists():
        print("Loading existing gene matrices...")
        transition_matrix = np.load("g_final_transition_matrix.npy")
        emission_matrix = np.load("g_final_emission_matrix.npy")
    elif test == "promoter" and Path("p_final_emission_matrix.npy").exists() and Path("p_final_transition_matrix.npy").exists():
        print("Loading existing promoter matrices...")  
        transition_matrix = np.load("p_final_transition_matrix.npy")
        emission_matrix = np.load("p_final_emission_matrix.npy")
    else:
        print("No existing matrices found, starting fresh...")
        for i in range(iterations):
            # estimate emission matrix
            transition_matrix, emission_matrix, _, log_likelihood = forward_backward(emission_num, emissions, states, transition_matrix, emission_matrix)
            
            if prev_log_likelihood is not None:
                delta = log_likelihood - prev_log_likelihood
                if np.isinf(log_likelihood):
                    print("Log-likelihood is -inf, stopping early")
                    break
                elif test == "gene" and abs(delta) <= 1e-3:
                    print(f"Log-likelihood converged (delta={delta:.5f}) at iteration {i}, stopping early")
                    break
                elif test == "promoter" and abs(delta) <= 1e-3:
                    print(f"Log-likelihood converged (delta={delta:.5f}) at iteration {i}, stopping early")
                    break

            prev_log_likelihood = log_likelihood

        if test == "gene":
            np.save("g_final_emission_matrix.npy", emission_matrix)
            np.save("g_final_transition_matrix.npy", transition_matrix)
        elif test == "promoter":
            np.save("p_final_emission_matrix.npy", emission_matrix)
            np.save("p_final_transition_matrix.npy", transition_matrix)

    # get final state predictions
    _, _, final_gamma, _ = forward_backward(emission_num, emissions, states, transition_matrix, emission_matrix)
    
    pos_probs = final_gamma[:, 0]
    if test == "gene": # top 50000
        top_genes = np.argpartition(pos_probs, -50000)[-50000:] + 1
    elif test == "promoter": # top 5000
        top_genes = np.argpartition(pos_probs, -5000)[-5000:] + 1
    return top_genes 

def predict(top_genes, top_promoters):
    # take top k by score first
    top_genes_sorted = sorted(top_genes)
    top_promoters_sorted = sorted(top_promoters)
    
    # write to file
    with open("predictions.csv", "w") as f:
        for interval in top_genes_sorted:
            f.write(f"G, {interval}\n")  
        for interval in top_promoters_sorted:
            f.write(f"P, {interval}\n")
        print("Predictions written to predictions.csv")
    
    return

# main------------------------------------------------------------------------------------

# initialization 1 
# g_emission_matrix = np.array([[0.25, 0.25, 0.35, 0.15], [0.05, 0.05, 0.05, 0.85]])
# p_emission_matrix = np.array([[0.3, 0.3, 0.3, 0.1], [0.05, 0.05, 0.05, 0.85]])
# g_transition_matrix = np.array([[0.995, 0.005], [0.005, 0.995]])
# p_transition_matrix = np.array([[0.9, 0.1], [0.01, 0.99]])

# initialization 2 
# g_emission_matrix = np.array([[0.25, 0.25, 0.35, 0.15], [0.05, 0.05, 0.05, 0.85]])
# p_emission_matrix = np.array([[0.15, 0.15, 0.6, 0.1], [0.2, 0.2, 0.1, 0.5]])
# g_transition_matrix = np.array([[0.995, 0.005], [0.005, 0.995]])
# p_transition_matrix = np.array([[0.999, 0.001], [0.05, 0.95]])

# initialization 3 - based on observed frequencies
g_emission_matrix = np.array([[0.25, 0.25, 0.15, 0.35], [0.037, 0.038, 0.016, 0.91]])
p_emission_matrix = np.array([[0.15, 0.10, 0.65, 0.10], [0.026, 0.01, 0.012, 0.952]])
g_transition_matrix = np.array([[0.99, 0.01], [0.01, 0.99]])
p_transition_matrix = np.array([[0.95, 0.05], [0.001, 0.999]])

print("Initializing gene marks...")
gene_marks = initialize("gene_marks.fasta")

print("Initializing promoter marks...")
promoter_marks = initialize("promoter_marks.fasta")

top_genes = baum_welch(gene_marks, "xyzn", ["gene", "non-gene"], g_transition_matrix, g_emission_matrix, iterations=30, test="gene")
print(f"top gene probs: {top_genes[:5]}")

top_promoters = baum_welch(promoter_marks, "xyzn", ["promoter", "non-promoter"], p_transition_matrix, p_emission_matrix, iterations=30, test="promoter")
print(f"top promoter probs: {top_promoters[:5]}")

predict(top_genes, top_promoters)

with zipfile.ZipFile("predictions.zip", "w") as zipf:
    zipf.write("predictions.csv")
