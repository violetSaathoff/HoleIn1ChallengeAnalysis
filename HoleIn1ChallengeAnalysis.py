# -*- coding: utf-8 -*-
"""
Created on Fri May 16 20:53:36 2025

@author: violet
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from functools import cache
from scipy.optimize import curve_fit
from scipy.stats import norm
from numpy.random import random, normal

"""parameters"""
courses = ['bigputts', 'swingtime', 'teeaire', 'waukesha', 'gastraus']
dataset = courses[-1]
warmup_days = 0 #  how many days of data at the start of the challenge should be ignored as "warm up"
weight_spread = 0.5
use_weights = 0
min_HI1_probability = 0
target = 29
z_score_target = target + 0.5
use_weights = use_weights and weight_spread > 0

"""data"""
def readlines(filepath:str) -> list:
    lines = None
    with open(filepath, 'r') as f:
        lines = f.readlines()
        f.close()
    return [l[:-1] if l.endswith('\n') else l for l in lines]

def writelines(filepath:str, lines:list) -> None:
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
        f.close()

def load(course:str = dataset) -> list:
    return [[int(s) for s in l.split(': ')[-1].split(',')] for l in readlines(f"{course}.txt")]

raw = load()

current_day_front_half = []
if raw and len(raw[-1]) == 9:
    current_day_front_half = raw.pop()

def save(course:str = dataset, data:list = raw) -> None:
    n = len(str(len(data)))
    lines = []
    for day, scores in enumerate(data, start  = 1):
        day = str(day)
        while len(day) < n: day = ' ' + day
        lines.append(f"{day} : " + ', '.join(','.join(map(str, scores[i:i + 9])) for i in [0, 9]))
    if len(current_day_front_half) == 9:
        day = str(len(lines) + 1)
        while len(day) < n: day = ' ' + day
        lines.append(f"{day} : " + ','.join(map(str, current_day_front_half)))
    writelines(f"{course}.txt", lines)

data = np.array(raw[warmup_days:], int)

"""pre-processing"""
total_shots = np.sum(data, axis = 1)
total_shots_by_half = (np.sum(data[:,:9], axis = 1), np.sum(data[:,9:], axis = 1))
if current_day_front_half:
    total_shots_by_half = (np.hstack((total_shots_by_half[0], sum(current_day_front_half))), total_shots_by_half[1])

days = len(data)
min_score = int(np.min(np.sum(data, axis = 1)))
max_score = int(np.max(np.sum(data, axis = 1)))
min_score_possible = int(np.sum(np.min(data, axis = 0)))
max_score_possible = int(np.sum(np.max(data, axis = 0)))

# choose a method for tallying the shots
weights = None # used when plotting histograms
if not use_weights:
    # weight all days the same
    counts = [Counter(data[:,i]) for i in range(18)]
    for d, s in enumerate(current_day_front_half):
        counts[d][s] += 1
else:
    # weight more recent days more
    counts = [defaultdict(int) for i in range(18)]
    extra = int(len(current_day_front_half) > 0)
    weights = np.linspace(1 - weight_spread, 1 + weight_spread, days + extra)
    for i, w in enumerate(weights[:-1] if extra else weights):
        for h in range(18):
            counts[h][data[i][h]] += w
    for d, s in enumerate(current_day_front_half):
        counts[d][s] += weights[-1]
    if extra: weights = weights[:-1]


# Compute the Shot Probabilities for Each Hole Directly (this is the main tool I use later)
sums = [sum(hole.values()) for hole in counts]  #  should equal [days, days, ..., days], but this is safer in case I do weird weighting shit
shot_probabilities = [[hole[s] / n for s in range(1, max(hole.keys()) + 1)] for hole, n in zip(counts, sums)]


# Assert that the Probability of a Hole-In-1 is Never 0
if min_HI1_probability > 0:
    for hole in shot_probabilities:
        if hole[0] < min_HI1_probability:
            # Adjust the Hole-In-1 Probability
            hole[0] = min_HI1_probability * 1 / (1 - min_HI1_probability*len(hole))
            
            # Normalize the Probability Distribution for the Hole
            s = sum(hole)
            for i in range(len(hole)):
                hole[i] /= s


# Compute the Expected Number of Shots for Each Hole (this isn't really used, but it's still fun to see!)
expected_shots = [sum(p*t for t, p in enumerate(hole, start = 1)) for h, hole in enumerate(shot_probabilities, start = 1)]


"""utility functions"""
# Compute the Product of an Iterable
def product(iterable:iter):
    prod = 1
    for x in iterable:
        prod *= x
    return prod

# Compute the Probability a Specific Path Occured
def path_probability(path:list, start:int = 0):
    return product((shot_probabilities[h][s - 1] for h, s in enumerate(path, start = start)))

def gaussian(x, mu:float, sigma:float):
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def quadratic(a:float, b:float, c:float):
    d = np.sqrt(b**2 - 4*a*c)
    return (-b - d)/(2*a), (-b + d)/(2*a)

def estimate_single_player_probabilities(probabilities = shot_probabilities, plot:bool = True):
    def get_hole_probabilities(q_hole):
        P = []
        S = 0
        for q in reversed(q_hole):
            a, b = quadratic(1, 2*S, -q)
            p = a if 0 <= a <= 1 else b
            P.append(p)
            S += p
        return P[::-1]
    
    P_single = list(map(get_hole_probabilities, probabilities))
    
    if plot:
        plot_hole_probabilities(P_single)
    
    return P_single

"""analysis methods"""
 
@cache
def P_success(hole:int = 0, shots:int = target, exact:bool = False, end = 18):
    """Estimate the Likelihood of Beating the Target by Trying All Possible Options
    
    THIS IS THE MAIN METHOD FOR MY ANALYSIS 
    
    It recursively searches the entire probability tree to figure out the probability they'll complete 
    all the holes starting with hole <hole> and ending with hole <end - 1> (holes are 0-indexed, 
    so they go from 0-17, not from 1-18), given <shots> total shots. If <exact> == True, it computes the 
    probability they'll use exactly all of the shots they've been given (i.g. "what is the probability they 
    shoot a 29"), and if it's false it computes the probability they'll use complete the holes with shots 
    remaining (i.e. "what is the probability they shoot a 29 or better").
    
    To compute probabilities for the back 9, use calls like P_success(9, 29 - x), where x is their score on 
    the front 9. Similarly, to compute probabilities for the front 9, use calls like P_success(shots = x, end = 9) 
    where x is the number of shots they're allowed on the front 9. By similar calls you can compute probabilities 
    for any arbitrary stretch of holes.
    
    A more "math" version of what it's computing is:
        P(score == <shots> over holes [<hole>, <end>))  if exact
        P(score <= <shots> over holes [<hole>, <end>))  else
    where holes are 0-indexed
    
    This function uses the recursion relation:
        P_success(h, s) = P(1 on hole h) * P_success(h, s - 1) + P(2 on hole h) * P_success(h, s - 2)) + ...
    (which is kinda just how probability trees work lol)
    And the base case:
        P_success(end, shots, True, end) = 1 if shots == 0 else 0
    (i.e. if you have no holes remaining and no shots remaining, then you've completed all the holes 
    using exactly the specified number of shots. And otherwise, you haven't)
    
    This method is cached to improve speed, and prunes where possible (i.e. if a computation is easily 
    known to result in 0 contribution to the final result it is skipped). The cache's hit rate isn't 
    that high when it first computes the tree, but without caching this computation takes a very long 
    time to complete (and with it it's almost instantaneous, running in under 1ms on my computer).
    """
    
    # define/use a recursive helper (preserves the documentation/help functions in the IDE)
    if shots < end - hole:
        # base case: only getting holes-in-one is no longer good enough, so the probability is 0
        return 0
    elif not exact:
        # P(total_shots <= <shots>) = P(total_shots = 18) + P(total_shots = 19) + ... + P(total_shots = <shots>)
        # Doing it this way, rather than by editing the base case saves cache space and computation time by about a factor of 2
        # (which doesn't really matter, but still, why not lol)
        return P_success(hole, shots - 1, False, end) + P_success(hole, shots, True, end)
    elif hole == end:
        # base case: there are no more holes to play -> return if they have no extra shots remaining
        return shots == 0
    else:
        # recursive case -> sum the contribution from getting a every possible score on the current hole
        return sum(p * P_success(hole + 1, shots - s, exact, end) for s, p in enumerate(shot_probabilities[hole], start = 1) if p > 0)

def P_success_given_front(front_score:int):
    """Compute the Probability of Completing the Challenge Given a Score on the Front 9"""
    return P_success(9, target - front_score, False, 18)

# Generate All Possible "Paths" to Beating the Challenge
def Paths(shots:int = target, hole:int = 0, end = 18, exact:bool = False, compute_probabilities:bool = True) -> list:
    """Generate All Possible "Paths" to Beating the Challenge
    
    This effectively does the same computation as P_success(), except that it tracks the paths, and can't be cached. 
    This means it's significantly slower, so it is not used in the main analysis, but can be used to compute interesting 
    metrics like "what score to they realistically need on the front 9 to have good odds on the day". That being said, 
    those sorts of metrics can also be computed using P_success() if you're clever.
    """
    
    # a recursive helper function which walks along all possible paths
    start = hole
    path = []
    def helper(hole:int, shots:int, P:float):
        if hole == end:
            if P > 0 and shots == 0 or (not exact and shots > 0):
                yield path_probability(path, start) if compute_probabilities else path.copy()
        elif P > 0 and shots >= end - hole:
            for s, p in enumerate(shot_probabilities[hole], start = 1):
                path.append(s)
                yield from helper(hole + 1, shots - s, P * p)
                path.pop()
    
    # use the helper
    yield from helper(hole, shots, 1.0)

def montecarlo(N:int = 1000, shots:int = target, gaussian:bool = True, output:int = 0):
    """Simulate N Rounds of Minigolf to Estimate the Rate of Shooting <shots> or Better 
    (either modeling each hole as a guassian, or by using 'shot_probabilities')
    
    In practice, P_success() should always give more accurate results (and run faster); 
    and as a result, this is not used in my main analysis.
    """
    
    # Define a Sample Function
    warning = None
    if gaussian:
        # Get the (Weighted) Probability Distributions for Each Hole
        W = np.array([[w for _ in range(18)] for w in weights]) if type(weights) == np.ndarray else 1
        means = np.mean(data * W, 0)
        devs = np.std(data, 0)
        
        # Define the Sample Function
        def r_shots(hole:int):
            return max(int(round(normal(means[hole], devs[hole]))), 1)
    else:
        # Compute the Cumulative Probability Distribution for Each Hole
        cdfs = [np.cumsum(hole) for hole in shot_probabilities]
        
        # Define the Sample Function
        warning = ['warning: bad distributions encountered']
        def r_shots(hole:int):
            r = random()
            for s, p in enumerate(cdfs[hole], start = 1):
                if r < p:
                    return s
            print(warning[0], end = '') # warning
            warning[0] = '.'
            return s + 1
    
    # Run the Simulation N Times
    S = [sum(map(r_shots, range(18))) for _ in range(N)]
    mean = np.mean(S)
    dev = np.std(S)
    z = (shots + (z_score_target - target) - mean) / dev
    p = sum(s < 30 for s in S) / len(S)
    q = norm.sf(-z)
    
    if type(warning) == list and warning[0] == '.': print('\n\n')
    
    # Print the Results
    if output == 0:
        print(f"Win Probability: {p}")
        if p > 0: print(f"Expected Days: {1/p}")
        print(f"Average Shots: {mean} +/- {dev}")
        print(f"Z-Score: {z}")
        print(f"Alt Win Probability: {q}")
        if q > 0: print(f"Alt Expected Days: {1/q}")
    elif output == 1:
        return S
    elif output == 2:
        return mean, dev, p

def hole_in_one_analysis(print_results:bool = True):
    # Count the Holes in X
    counts = [Counter(np.sum(data == x, axis = 1)) for x in range(1, np.max(data) + 1)]
    for score, count in Counter(current_day_front_half).items():
        counts[score - 1][count] += 1
    
    # compute the probability of shooting X Y's (e.g. probability of shooting 7 1's)
    @cache
    def Q(count, score):
        return counts[score - 1][count] / sum(counts[score - 1].values())
    
    # a helper to find all the options
    @cache
    def P(i:int = 0, holes:int = 18, shots:int = target, exact:bool = False):
        score = i + 1
        if holes < 0 or shots < holes * score:
            # they don't have enough shots left to break 30
            # "if on every hole remaining they shoot <score>, they'll have shot <holes>*<score> shots"
            # "they can't do better, because the next scores in 'counts' are all higher"
            # "so if the number of shots they have remaining isn't enough, they can't break 30"
            return 0
        elif not exact:
            # use the cumulative distribution
            return P(i, holes, shots, True) + P(i, holes, shots - 1, False)
        elif holes == 0:
            # they broke 30, but did they do it exactly
            return int(shots == 0)
        elif i < len(counts):
            # recursive Case
            return sum(Q(count, score) * P(score, holes - count, shots - count * score, exact) for count in counts[i])
        else:
            # we've run out of score options without shooting a full round
            return 0
    
    # Use the Recursive Solver to Compute the Probability of Breaking 30
    p = P()
    
    """
    The 'exact' probabilities don't sum to  1, and I think this is because not every combination of 
    scores leads creates a valid path to 18 holes due to the sparcity of the data. Meaning that although 
    the prior distributions used for each score all sum to 1, parts of those distributions get 'dropped'. 
    Dividing by this factor seems to produce good results (i.e. consistent with results generated using 
    other methods), but I do not have a good justification for this; and it does call into question the 
    validity of the main result (p) as estimated by this method.
    """
    s = sum(P(shots = score, exact = True) for score in range(min_score_possible, max_score_possible + 1))
    
    # Use the Recursive Solver to Compute the Expected Number of Shots
    e = sum(P(shots = score, exact = True) * score / s for score in range(min_score_possible, max_score_possible + 1))
    v = sum(P(shots = score, exact = True) * score**2 / s for score in range(min_score_possible, max_score_possible + 1))
    v = np.sqrt(v - e**2)
    
    # print or return the result
    if print_results:
        print(f'Probability = {p:.5f}')
        print(f'Expected Days: {1/p:.1f}')
        print(f'Expected Score: {e:.2f} ± {v:.2f}')
    else:
        return p, e, v

def rank_holes(expectation_value:bool = False, print_results:bool = True, plot:bool = True):
    """Rank the Holes Based on how 'Good' they are (either by hole-in-1 probability, or by expected shots)"""
    expectation_value = int(bool(expectation_value))
    holes = [(h, P[0], e) for h, (P, e) in enumerate(zip(shot_probabilities, expected_shots), start = 1)]
    holes.sort(key = lambda x : x[2 - expectation_value], reverse = expectation_value)      #  secondary sort first
    holes.sort(key = lambda x : x[1 + expectation_value], reverse = 1 - expectation_value)  #  primary sort
    
    if plot:
        fig, ax = plt.subplots()
        x = list(range(1, 19))
        xlabels = [h for h, _, _ in holes]
        y = [(e if expectation_value else p) for _, p, e in holes]
        plt.bar(x, y, color = 'violet') 
        plt.xticks(x, xlabels)
        plt.xlabel('Hole')
        plt.ylabel('Expected Shots' if expectation_value else 'Hole-In-One Probability')
        if not expectation_value: plt.ylim(0, 1)
        plt.show()
    
    if print_results:
        print('Hole Ranking: (Hole-In-One Probability, Expected Shots)')
        for h, p, e in holes:
            print(f"{h}\t:\t{100*p:.1f}%\t({e:.2f})")
    else:
        return holes

def plot_round_probability(scores:list = 0, starting_hole:int = 1):
    """Plot the Probability of Success Before Each Shot Throughout a Round (<scores> can also be the 1-indexed day number: 0 -> most recent round)"""
    if type(scores) == int:
        scores = data[scores - 1]
    
    # Get the Probabilities
    X = []
    P = []
    S = 0
    for h, s in enumerate(scores, start = starting_hole):
        X.append(h)
        P.append(P_success(h - 1, target - S))
        S += s
    
    # Plot the Result
    fix, ax = plt.subplots()
    plt.errorbar(X, P, fmt = '.', color = 'violet')
    plt.plot(X, P, color = 'violet')
    plt.ylabel(f'Probability of Breaking {target + 1}')
    plt.xlabel('Hole')
    plt.xticks(X, X)
    plt.show()

def plot_hole_probabilities(probabilities:list = shot_probabilities, ranked:bool = True, colors = ['violet', 'mediumorchid', 'purple', 'indigo'], width = 0.75):
    """Plot the Hole Probabilities as a Stacked Bar Plot"""
    
    # get the data in the proper format
    y_bins = max(map(len, probabilities))
    if ranked:
        ranked = sorted([(x, P) for x, P in enumerate(probabilities, start = 1)], key = lambda x : x[1][0], reverse = True)
        x = [x for x, _ in ranked]
        probabilities = [np.array([hole[s] if s < len(hole) else 0 for _, hole in ranked]) for s in range(y_bins)]
    else:
        x = list(range(1, 19))
        probabilities = [np.array([hole[s] if s < len(hole) else 0 for hole in probabilities]) for s in range(y_bins)]
    
    # Make the Main Figure
    xx = list(range(1, 19))
    bottom = np.zeros(18)
    fig, ax = plt.subplots()
    for s in range(y_bins):
        ax.bar(xx, probabilities[s], width, label = str(s + 1), bottom = bottom, color = colors[s])
        bottom += probabilities[s]
    
    # Add the Text/Legend
    ax.set_xlabel('Hole')
    ax.set_ylabel('Probability')
    ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(xx, x)
    
    # Show the Figure
    plt.show()

def plot_expected_shots(cumulative:bool = False):
    """Either Plots the Expected Number of Shots for Each Hole, 
    or the Expected Number of Total Shots After Each Hole"""
    fig, ax = plt.subplots()
    x = list(range(1, 19))
    if cumulative:
        y = np.cumsum(expected_shots)
        plt.plot(x, y, color = 'black')
        plt.ylabel('Total Shots')
    else:
        y = expected_shots
        plt.errorbar(x, y, fmt = '_', color = 'black')
        plt.ylabel('Shots')
    plt.xlabel('Hole')
    plt.xticks(x, x)
    plt.ylim(1, round(2 * max(y) + 1) / 2)
    plt.show()

def plot_halves(return_data:bool = False, print_results:bool = True):
    """Plots the Likelihood of Various Scores on Either the Front 9 or the Back 9"""
    
    # Compute the Max Score for Either Half
    endpoints = [(0, 9), (9, 18)]
    max_score = max(sum(map(len, shot_probabilities[start:end])) for start, end in endpoints)
    
    # Get the Data
    x = list(range(9, max_score + 2))
    P = [[P_success(start, s, True, end) for s in x] for start, end in endpoints]
    
    # Plot the Results
    fig, ax = plt.subplots()
    for i, (label, color) in enumerate(zip(['Front 9', 'Back 9'], ['violet', 'teal'])):
        plt.plot(x, P[i], color = color, label = label)
    plt.xlabel("Score")
    plt.ylabel('Likelihood')
    plt.legend()
    plt.show()
    
    # Calculate and Print Some Stats
    means = [np.mean(half) for half in total_shots_by_half]
    devs = [np.std(half) for half in total_shots_by_half]
    s = 0
    u = 0
    for half, mean, dev in zip(['Front', 'Back'], means, devs):
        if print_results: print(f"{half} Half: {mean:.2f} ± {dev:.2f} shots")
        s += mean
        u += dev**2
    u = np.sqrt(u)
    z = (z_score_target - s) / u
    p = norm.sf(-z)
    if print_results: print(f"Overall: {s:.2f} ± {u:.2f} shots -> z = {z:.3f} -> p = {p:.5f} -> Exp Days = {1/p:.1f}")
    
    # Return the Data
    if return_data:
        return P, u, s, z, p

def analyze_front_vs_back():
    """Generates a Plot Showing What Score Needs to be Achieved on the Front 9
    
    This multiplies the probability of scoring exactly S on the front half by 
    the probability of beating the target (29) given a score S on the front half, 
    and plots the results for all possible scores on the front half.
    """
    
    # Get the Data
    x = list(range(9, target - 9))
    probabilities = [P_success(0, s, True, 9) * P_success(9, target - s, False, 18) for s in x]
    
    # Plot the Result
    fig, ax = plt.subplots()
    plt.plot(x, probabilities, color = 'violet')
    plt.ylabel('Overall Likelihood of Success')
    plt.xlabel('Score on the Front 9')
    plt.show()
    
    # Return the Data
    return probabilities

def plot_P_success_given_front(front_scores = range(9, target - 9)):
    """Generate a Plot Showing the Probability of Completing the Challenge Given a Score on the Front 9"""
    # Get the Data
    P = [P_success(9, target - s, False, 18) for s in front_scores]
    
    # Plot the Result
    fig, ax = plt.subplots()
    plt.plot(front_scores, P, color = 'violet')
    plt.ylabel('Likelihood of Success')
    plt.xlabel('Score on the Front 9')
    plt.show()
    
    # Return the Data
    return P

def all_plots(score_histogram:bool = False, round_shots:list = 0):
    plot(7 if score_histogram else 3)
    # rank_holes()
    # plot_expected_shots()
    plot_halves(print_results = False)
    analyze_front_vs_back()
    plot_P_success_given_front()
    plot_hole_probabilities()
    plot_round_probability(round_shots)

"""Run the Main Analysis"""

def main_analysis():
    # Compute the Round Score Distribution Using the Recursive Solver
    X = np.array(range(3*18 + 1))
    exact_probabilities = np.array([P_success(shots = s, exact = True) for s in X])
    probabilities = np.array([P_success(shots = s) for s in X])
    expected_score = sum(s*p for hole in shot_probabilities for s, p in enumerate(hole, start = 1))
    expected_var = sum(sum(s*s*p for s, p in enumerate(hole, start = 1)) - sum(s*p for s, p in enumerate(hole, start = 1))**2 for hole in shot_probabilities)
    expected_stdev = np.sqrt(expected_var)
    z0 = (z_score_target - expected_score) / expected_stdev
    p0 = norm.sf(-z0)
    
    # Do Gaussian Fits
    params, var = curve_fit(gaussian, np.array(range(0, len(exact_probabilities))), exact_probabilities)
    var = np.sqrt(np.diag(var))
    mean_shots = np.mean(total_shots * (weights if type(weights) == np.ndarray else 1))
    std_shots = np.std(total_shots)
    
    # Mean/Dev of Scores by Half Analysis
    e5 = sum(np.mean(half) for half in total_shots_by_half)
    u5 = np.sqrt(sum(np.std(half)**2 for half in total_shots_by_half))
    z5 = (z_score_target - e5) / u5
    p5 = norm.sf(-z5)
    
    # Do a Hole-In-1 Count Analysis
    p4, s4, v4 = hole_in_one_analysis(False)
    
    # Run a Quick Monte-Carlo Sim (using a Gaussian Fit for Each Hole)
    mean, dev, p3 = montecarlo(N = 1000, gaussian = True, output = 2)
    
    # Compute the Z-Scores and P Values for the Fits (assuming anything less than 29.5 "breaks 30")
    z1 = (z_score_target - mean_shots) / std_shots
    z2 = (z_score_target - params[0]) / params[1]
    p1 = norm.sf(-z1)
    p2 = norm.sf(-z2)
    
    # Get the Expected Number of Days From a P Value
    def E(p:float) -> str:
        return 'NaN  ' if p <= 0 else f"{1 / p + warmup_days:.2f}"
    
    # Print the Results
    print('\n'.join([
        'Method\t\t\tMean\t\tSTDev\t\tZ-Score\t\tProbability\t\tExpected Days', 
        '-'*81,
        f'Tree Search\t\t{expected_score:.2f}\t\t \t\t\t\t\t\t  {probabilities[target]:.5f}\t\t\t{E(probabilities[target])}',
        '-'*81,
        f'Tree Search*\t{expected_score:.2f}\t\t{expected_stdev:.3f}\t\t{z0:.3f}\t\t  {p0:.5f}\t\t\t{E(p0)}',
        f'Gaussian Fit*\t{params[0]:.2f}\t\t{params[1]:.3f}\t\t{z2:.3f}\t\t  {p2:.5f}\t\t\t{E(p2)}',
        f'Round Scores\t{mean_shots:.2f}\t\t{std_shots:.3f}\t\t{z1:.3f}\t\t  {p1:.5f}\t\t\t{E(p1)}',
        f'Half Scores\t\t{e5:.2f}\t\t{u5:.3f}\t\t{z5:.3f}\t\t  {p5:.5f}\t\t\t{E(p5)}',
        #f'HIO Analysis\t{s4:.2f}\t\t{v4:.3f}\t\t\t\t\t  {p4:.5f}\t\t\t{E(p4)}',
        f'Monte Carlo\t \t{mean:.2f}\t\t{dev:.3f}\t\t\t\t\t  {p3:.5f}\t\t\t{E(p3)}',
        '*based on Tree Search results'
    ]))
    
    # Print the Expected Shots
    f = sum(expected_shots[:9])
    b = sum(expected_shots[9:])
    t = sum(expected_shots)
    easier = 'Front' if f < b else 'Back'
    print(f'\nExpected Shots: {f:.2f} (Front) + {b:.2f} (Back) = {t:.2f} (Total)')
    print(f'{easier} is {abs(f - b):.2f} shots easier')
    
    # Print the Odds of Completing the Challenge on the Current Dat
    if current_day_front_half:
        score = sum(current_day_front_half)
        p = P_success(9, target - score)
        print(f'\nProbability of Winning Today (Sitting on {score}): {100*p:.2f}%')
    else:
        print(f'\nLikelihood of Previous Round: {100*path_probability(data[-1]):.3f}%')
    
    if False:
        r = np.corrcoef(list(range(len(total_shots))), total_shots)
        print(f'\nTotal Shots vs Time Correlation: {r[0,1]:.3f}')
    
    # Print Some Notes on the Methodology
    if False:
        print('\n'.join([ 
            '',
            'The main method is the Tree Search method, which uses the estimated shot probabilities to compute the ',
            'likelihood of every possible round. These results can be used in a few other ways though. First, is ',
            'the Tree Search*, where the probabilities are computed for exact scores using the Tree Search, and ',
            'then expectation values are calculated for the round score and its variance, which are then used to ',
            'compute a Z-score, which is converted to a probability using a 1-tailed test. The Gaussian Fit method ',
            'works very similarly, but instead of computing expectation values, the exactt probabilities are fit to ',
            'a Gaussian probability density function, resulting in slightly different estimates for the mean and ',
            'standard deviation. The Round Scores method, on the other hand, just computes the mean and standard ',
            'deviation of the round scores directly, and then uses the same 1-tailed test to estimate the probability. ',
            'Lastly, the Monte Carlo method simulates 1000 rounds where each hole is approximated with a Gaussian fit. ',
            'It is possible to change the parameters of the Monte Carlo simulation to instead use the measured shot ',
            'probabilities, and it is possible to run an arbitrarily large number of simulated rounds, but I think ',
            'this is a decent starting place (which doesn\'t take too much time to run) - though repeated runs do ',
            'give fairly different results. The HIO Analysis tries to estimate a distribution for "how likely are ',
            'they to shoot X 1\'s in a round?" (and for 2\'s, 3\'s, etc.), and then tries all combinations of those ',
            'to estimate the probability they shoot a combination which breaks 30. This method currently suffers from ',
            '\'gaps\' in the data, and does need to use a fudge factor to produce sensible results for the mean and ',
            'standard deviation; which does call into question the validity of it\'s main result.'
        ]))
    
    # Plot the Probability Density Functions
    def plot(included:int = 3, x_min:int = min_score_possible - 2, x_max:int = max_score_possible + 2):
        """Plot the Probability Density Function and Cumulative Density Function Generated by the Solver, and a histogram of the actual shot totals
        
        <included> is a 3-bit bitmask: <histogram><cdf><pdf> which determines which parts of the plot are included. 
        The default is 3 = 011 = "no histogram, yes CDF, yes PDF"
        """
        x_max += 1
        plt.figure()
        if included&1: plt.plot(X[x_min:x_max], exact_probabilities[x_min:x_max], label = 'P(score = x)', color = 'violet')
        if included&2: plt.plot(X[x_min:x_max], probabilities[x_min:x_max], label = 'P(score <= x)', color = 'teal')
        if included&4: plt.hist(np.sum(data, axis = 1), [x - 0.5 for x in range(x_min, x_max + 2)], density = True, weights = weights)
        plt.ylabel('Probability')
        plt.xlabel('Shots')
        if included not in (1, 2, 4): plt.legend()
        plt.show()
    
    # Make the Plots
    plot(3)
    # rank_holes()
    # plot_expected_shots()
    plot_halves(print_results = False)
    analyze_front_vs_back()
    plot_P_success_given_front()
    plot_hole_probabilities()
    plot_round_probability()
    
    # Return the Data and the Plot Function
    return probabilities, exact_probabilities, plot

probabilities, exact_probabilities, plot = main_analysis()
