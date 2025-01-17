'''
Helper functions to calculate the equitable
cooperation measure
'''

import numpy as np


def eq_coop(T:int, mu:float, scores:list) -> float:
    '''
    Calculates the equitable cooperation measure.
    
    Input:
        - T, the number of rounds
        - mu, the threshold
        - scores, a list of scores
        
    Output:
        - equitable cooperation measure of the scores
    '''

    '''
    Helper functions
    '''

    # Get number of playes from scores
    N = len(scores)
    # calculate square distance between scores and mu
    dist = calculate_dist_F(N, mu, scores)
    # calculate min distance in a perfect equitable cooperation scenario
    zeta_bound = calculate_zeta_bound_F(N, mu, T)
    # Return formula for equitative cooperation
    return 1 - (dist - zeta_bound) / (1 + mu - zeta_bound)


def calculate_dist_F(N:int, mu:float, scores:list) -> float:
    '''
    Calculates the squared distance between the scores and the threshold.
    
    Input:
        - N, number of players
        - mu, threshold
        - scores, list of scores
        
    Output:
        - distance (float)
    '''
    return np.sqrt(np.sum(np.square(scores - mu)) / N)



def calculate_zeta_bound_F(N:int, mu:float, T:int) -> float:
    '''
    Determines the minimum distance from all players' score to the threshold 
    in a scenario with perfect equitable cooperation. This situation occurs because
    the threshold might not give rise to all players getting the fair quantity 
    in the given number of rounds. Some players will get a payoff a bit higher 
    and others a bit lower than the threshold.
    
    Input:
        - N, number of players
        - mu, threshold
        - T, number of rounds
        
    Output:
        - minimum euclidean distance to the threshold in a perfect equitable cooperation scenario.
    '''
    
    # Fair quantity
    F = int(np.floor(mu * T))
    # Bar's capacity
    C = int(np.floor(mu * N))
    if C == 0:
        # print(f'Warning, C = 0 with N:{N} --- mu:{mu} --- T:{T}')
        return mu
    # Number of rounds for all players with fair quantity
    P = int(np.ceil(N / C))
    # number of over-receivers
    H = calculate_H(N, T, C, P)   
    # number of under-receivers
    L = N - H
    assert(L >= 0), f'N:{N} --- mu:{mu} --- T:{T}'
    if F == mu * T:
        high_payoff = F / T
        low_payoff = (F - 1) / T
    else:
        high_payoff = (F + 1) / T
        low_payoff = F / T        
    return np.sqrt((H * ((high_payoff - mu) ** 2) + L * ((low_payoff - mu) ** 2)) / N)


def calculate_H(N:int, T:int, C:int, P:int) -> int:
    '''
    Determines the number of players with high-payoff in a perfect equitable coordination situation.
    
    Input:
        - N, number of players
        - T, number of rounds
        - C, the bar's capacity as an integer
        - P, the minimum number of rounds for all players to get at least the fair score
    
    Output:
        - H, the number of players with the high-payoff
    '''
    if C == 0:
        return None
    # Chech excess capacity in two rounds
    if 2 * C > N:
        # number of over-receivers
        O = 2 * C - N
    else:
        # number of over-receivers
        O = N % C
    # number of players with high payoff
    H = O if T % P == 0 else C * (T % P) 
    return H


