# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:05:05 2025

@author: Gabriel Chew 
"""
import numpy as np


class UndefinedError(Exception):
    pass

class NotEnoughMatricesError(Exception):
    pass


def nonzero_for_reverse(P):
    """
    
    Parameters
    ----------
    P : np.array
        The transition matrices of the Markov chain. Can be time-homogeneous or non-time-homogeneous.
        IMPORTANT: We assume that P(0) does not have zero columns. 

    Returns
    -------
    entries_to_fill : list
        A minimal list of the indices in the initial distribution that must be non-zero for the 
        reverse process to be well-defined. 

    """
    if ~(P[0].sum(axis=0).all()):
        full_zero_columns = P[0].sum(axis=0)[P[0].sum(axis=0) == 0]
        raise UndefinedError(f"The algorithm does not work on this matrix, as column(s) {full_zero_columns}) are fully zero.")
    
    
    zero_cols = np.array([True for i in range(len(P[0].T))]) 
    entries_to_fill = [] # The indices in the initial distribution that must be filled in. 


    """
    How the algorithm works:
        Step 1: Find the row with the most non-zero entries, r_1, and append it to entries_to_fill. 
        Step 2: Note down the entries in the row where the row was non-zero. 
            Justification - Assume that in the initial distribution, every entry other than r_1 is non-zero.
            Then mu(1) would have non-zero entries except where r_1 = 0. 
            To compensate for this, we need to search for the next row which has the most non-zero elements
            in the entries where r_1 was zero. 
        Step 3: Find the row with the most non-zero entries AMONG THE VALUES WHERE r_1 was zero. 
    """
        
    while zero_cols.any(): 
    #    """
    #    max_non_zeros = 0
    #    max_non_zeros_idx = None
    #    for key, val in matrix_columns.items():
    #        non_zeros = np.count_nonzero(val != 0)
    #       if non_zeros > max_non_zeros:
    #            max_non_zero = non_zeros 
    #            max_non_zero_idx = key
    #    """
    
        # Find all the non-zero entries in P(0), and obtain a row-wise sum of the number of non-zero elements.
        non_zeros = np.sum(P[0][:, zero_cols] != 0, axis=1) 
        
        # Find the row with the most non-zero columns 
        entries_to_fill.append(np.argmax(non_zeros)) 
        # Index the row with the most non-zero entries
        zero_cols = np.where((P[0][np.argmax(non_zeros)] == 0) & (zero_cols == True), zero_cols, False) 
        
    return entries_to_fill
    

def calc_reverse_process(P, initial_distribution, N): 
    """
    Parameters
    ----------
    P : numpy array. 
        An array of transition matrices corresponding to the transition matrix of each time-step in the chain.
    
    initial_distribution : numpy array. 
        The initial distribution of the Markov chain.
        
    N : int
        The number of time-steps before the chain terminates.

    Returns
    -------
    P_reverse_time: A (N, n, n) array consisting of the transition matrices P(n) for each n.
    
    distributions: A (N, n + 1) array consisting of the distributions (n) for each n.
    
    """
    
    # Multiply the intial distribution with the transition matrix P(0) to check that mu(1) has no non-zero entries,
    # to ensure that the reverse chain is well-defined.
    if ~((initial_distribution @ P[0]).all()): 
        print("Sorry, this transition matrix and initial distribution combination does not give a well-defined reverse chain.")
        raise UndefinedError
    
    num_states = len(P[0])
    P_reverse = np.zeros((N, num_states, num_states)) # Initialises the time-reversed matrices
    distributions = np.zeros((N + 1, num_states)) # Initialises the time-evolving distributions
    
    # We start with calculating the distributions by iteratively multiplying the distribution with the transition matrix 
    distributions[0] = initial_distribution
    for n in range(1, N + 1):
        distributions[n] = distributions[n - 1] @ P[n - 1]
        if ~(distributions[n].all()):
            print(f'This Reverse Chain is not well-defined: At time {n} there is a 0 entry in the distribution.')
    
    
    # Simply applying the reverse transition matrix formula with the broadcasting feature of numpy arrays
    for n in range(0, N): 
        P_reverse[n] = (P[N - n - 1].T * distributions[N - n - 1] ) / distributions[N - n].reshape((num_states,1))
        
    return P_reverse, distributions 


def calc_forward_comms(P, N, A, B):
    
    """
    Calculates the forward committors. 
    
    
    Parameters
    ----------
    P : np.array
        The transition matrices P(n) of the chain, with one transition matrix for each time.
    
    N : int
        The number of time-steps before the chain terminates.
        
    A: list of integers
        The elements of the starting set A.
        
    B: list of integers
        The elements of the ending set B.

    Returns
    -------
    committors: np.array
        An array consisting of the committor values for each element in the state space at each time.
    """


    num_states = len(P[0])

    
    # Initialise the committor object, and we have committors for n = 0 up to N, giving a total of N+1 rows
    committors = np.zeros((N + 1, num_states))  
    # Instantly calculate the committors at termination
    committors[N] = np.array([1 if i in B else 0 for i in range(num_states)])
    
    for n in range(N - 1, -1, -1):
        committors[n] =  P[n] @ committors[n + 1]
        committors[n, A] = 0
        committors[n, B] = 1
        
        #   if i == N:
        #       committor_n_1 = final_distribution
        #       committors[i] = committor_n_1
        #      continue
    #
       # committors[i] = committor_n
       # committor_n_1 = committor_n
    
    return committors


def calc_backward_comms(P_reverse, N, A, B):
    
    """
    Calculates the backward committors. 
    
    
    Parameters
    ----------
    P_reverse : np.array
        The transition matrices of the chain
    
    N : int
        The number of time-steps before the chain terminates.
        
    A: list of integers
        The elements of the starting set A.
        
    B: list of integers
        The elements of the ending set B.

    Returns
    -------
    committors: np.array
        An array consisting of the backward committor values for each element in the state space at each time.
    """
    
    if len(P_reverse) != N:
        raise UndefinedError("Something has gone wrong in your transition matrix.")
    
    num_states = len(P_reverse[0])
    
    # Initialise the committor object 
    committors = np.zeros((N + 1, num_states))  
    
    # Instantly set the committors value at the start
    committors[0] = np.array([1 if i in A else 0 for i in range(num_states)])
    
    for n in range(1, N + 1):
        committors[n] = P_reverse[N - n] @ committors[n - 1]
        committors[n, A] = 1 
        committors[n, B] = 0    
        
        #    if i == 0:  
        #       committor_1_n = initial_distribution
        #        committors[i] = committor_1_n
        #        continue
            
        # #    committor_n = P_reverse[N - i] @ committor[]
        # #    committor_n[A] = 1
        # #   committor_n[B] = 0
        # #    committors[i] = committor_n
        #     committor_1_n = committor_n
        
    return committors


def calc_reactive_dist(distributions, forward_comms, backward_comms):
    """
    

    Parameters
    ----------
    distributions : array 
        An array of size (N+1, num_states) consisting of the distributions mu(n) of the chain for each time.
    forward_comms : array
        An array of size (N+1, num_states) consisting of the forward committors of the chain for each time.
    backward_comms : TYPE
        An array of size (N+1, num_states) consisting of the backward committors of the chain for each time.

    Returns
    -------
    reactive_distributions : TYPE
        An array of size (N+1, num_states) consisting of the reactive distributions mu(n) of the chain for each time.

    """
    # Distributions, forward committors, and backward_committors are all the same length. We may
    # simply apply numpy elementwise multiplication features for arrays
    reactive_distributions = backward_comms * distributions * forward_comms 
    return reactive_distributions


def calc_cond_reactive_dist(reactive_distributions):
    """
    

    Parameters
    ----------
    reactive_distributions : array
        An array of size (N+1, num_states) consisting of the reactive distributions of the chain for each time.

    Returns
    -------
    cond_reactive_dist : TYPE
        An array of size (N+1, num_states) consisting of the conditional reactive distributions of the chain for each time. 
        NB: Entries 0 and N will be the zero vector because there are no reactive pathways at time 0 and at time N.

    """
    cond_reactive_dist = reactive_distributions.copy()
    
    # Dividing each element in reactive_distributions by its row sum. np.nan_to_num to output 0 in the nan columns.
    cond_reactive_dist = np.nan_to_num(reactive_distributions / np.sum(reactive_distributions, axis=1).reshape(len(reactive_distributions), 1))
    
    # for n in range(len(reactive_distributions)):
    #    cond_reactive_dist[n] = np.nan_to_num(reactive_distributions[n]/ (reactive_distributions[n]).sum())
    return cond_reactive_dist
    

def calc_prob_current(forward_comms, backward_comms, distributions, P):
    """
    

    Parameters
    ----------
    forward_comms : array
        An array of size (N+1, num_states) consisting of the distributions of the chain for each time.
    backward_comms : array
        An array of size (N+1, num_states) consisting of the distributionsof the chain for each time.
    distributions : array
        An array of size (N+1, num_states) consisting of the distributions of the chain for each time.
    P : array
        An array of size (N, num_states, num_states) consisting of the transition matrices of the chain for each time-step.

    Returns
    -------
    prob_current_matrix : array
        An array of size (N, num_states, num_states) consisting of the probability current of the chain for each time-step.

    """
    N = len(distributions) - 1
    num_states = len(P[0])
    prob_current_matrix = np.zeros((N, num_states, num_states)) # Initialise the probability current matrix
    
    # Applying the broadcast feature of numpy arrays and element-wise multiplication again. 
    # backward_comms[n] is reshaped into the same dimensions as P[n] with elements along rows all identical
    for n in range(N): 
        prob_current_matrix[n] =  backward_comms[n].reshape((num_states,1)) * P[n] * distributions[n].reshape((num_states,1)) * forward_comms[n+1]
    #for n in range(N):
    #    for i in range(num_states):
    #        for j in range(num_states):
    #            prob_currentent_matrix[n, i, j] = backward_comms[n, i] * P[i, j] * distributions[n, i] * q_plus[n + 1, j]
    
    return prob_current_matrix


def calc_cond_prob_current(prob_current):
    """
    

    Parameters
    ----------
    forward_comms : array
        An array of size (N+1, num_states) consisting of the distributions of the chain for each time.
    backward_comms : array
        An array of size (N+1, num_states) consisting of the distributionsof the chain for each time.
    P : array
        An array of size (N, num_states, num_states) consisting of the transition matrices of the chain for each time-step.


    Returns
    -------
    prob_current_matrix : array
        An array of size (N, num_states, num_states) consisting of the transition matrices of the chain for each time-step.

    """
    
    N = len(prob_current)
    num_states = len(prob_current[0])
    
    cond_prob_current_matrix = np.zeros((N, num_states, num_states)) # Initialise the probability current matrix
    prob_current_row_sum = np.sum(prob_current, axis=2)
    
    cond_prob_current_matrix = np.nan_to_num(prob_current / prob_current_row_sum.reshape(N, num_states, 1))
    
    return cond_prob_current_matrix

def calc_effec_current(prob_current): # Calculation of effective current
    """
    
    Parameters
    ----------
    prob_current : array
        An array of size (N, num_states, num_states) consisting of the probability current matrices of the chain for each time.
    

    Returns
    -------
    effec_current : array
        An array of size (N, num_states, num_states) consisting of the effective current matrices of the chain for each time-step.
    
    
    """

    effec_current = np.zeros(prob_current.shape)
    
    for n in range(len(prob_current)): 
        
        # Find where the P_ij- P_ji < 0, and replace with the number 0 
        effec_current[n] = np.where((prob_current[n] - prob_current[n].T) > 0, (prob_current[n] - prob_current[n].T), 0)

    return effec_current


def calc_rate(prob_current, A, B):
    """
    
    Parameters
    ----------
    prob_current : array
        An array of size (N, num_states, num_states) consisting of the probability current matrices of the chain for each time.
    A : list of ints
        The elements in the source set.
    B : list of ints
        The elements in the target set.
    

    Returns
    -------
    effec_current : array
        An array of size (N+1, num_states) that is non-zero only in the indices corresponding to elements in the source and target set.
    
    """

    N = len(prob_current) 
    num_states = len(prob_current[0])
    
    
    rates_A = np.zeros(N + 1)
    rates_B = np.zeros(N + 1) 
    
    
    rates_A[0:N] = prob_current.sum(axis=2)[:, A].flatten()
    rates_B[1:N+1] = prob_current.sum(axis=1)[:, B].flatten()
# =============================================================================
#     rates_A[0]= prob_current[0, A].sum() # Directly set the rate at A at time 0
#     rates_B[N] = prob_current[N - 1, :, B].sum() # Directly set the rate at B at time N.
#     
#     for n in range(1, N):
#         rates_A[n] = prob_current[n, A].sum() # Sum of outflow from A
#         rates_B[n] = prob_current[n - 1, :, B].sum() # Sum of inflow into B from previous timestep 
# =============================================================================
        
    return rates_A, rates_B
        
    
    
    
    
    
    
    
    