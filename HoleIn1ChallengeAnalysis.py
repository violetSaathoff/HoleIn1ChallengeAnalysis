# -*- coding: utf-8 -*-
"""
Created on Fri May 16 20:53:36 2025

@author: violet
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from functools import cache
from scipy.optimize import curve_fit
from scipy.stats import norm
from numpy.random import random, normal

"""-------------------- PARAMETERS --------------------"""
class params:
    courses = ['bigputts', 'swingtime', 'teeaire', 'swingtime waukesha', 'gastraus', 'bigputts waukesha']
    dataset = courses[-1]
    use_historical = 0 # False/True/2 (only affects supported courses)
    warmup_days = 0 #  how many days of data at the start of the challenge should be ignored as "warm up"
    weight_spread = 0.5
    use_weights = 0
    min_HI1_probability = 0
    target = 29
    z_score_target = target + 0.5
    use_weights = use_weights and weight_spread > 0
    bp18 = False # at big putts, when false, use hole 1 for hole 18, when true use the special hole for hole 18 (challenge days 11+)

"""-------------------- UTILITIES --------------------"""
# Compute the Product of an Iterable
def product(iterable:iter):
    prod = 1
    for x in iterable:
        prod *= x
    return prod

def gaussian(x, mu:float, sigma:float):
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def quadratic(a:float, b:float, c:float):
    d = np.sqrt(b**2 - 4*a*c)
    return (-b - d)/(2*a), (-b + d)/(2*a)

def readlines(filepath:str) -> list:
    """Read a Text File Line by Line"""
    lines = None
    with open(filepath, 'r') as f:
        lines = f.readlines()
        f.close()
    return [l[:-1] if l.endswith('\n') else l for l in lines]

def writelines(filepath:str, lines:list) -> None:
    """Write a Text File Line by Line"""
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
        f.close()

"""-------------------- DATA LOADING/SAVING/PRE-PROCESSING --------------------"""
class Round(list):
    """Data Ojbect for Storing Each Round's Info"""
    def __init__(self, line:str):
        self.raw = line
        self.name, scores = line.split(' : ')
        scores = scores.replace(' ', '').split(',')
        self.extend(int(s) for s in scores if s.isdigit())
        self.front = sum(self[:9]) if len(self) >= 9 else None
        self.back = sum(self[9:]) if len(self) == 18 else None
        self.completion = bool(self.front) + 2*bool(self.back)
        self.complete = self.completion == 3
        self.score = self.front + self.back if self.complete else None
    
    def __repr__(self) -> str:
        front = ','.join(map(str, (s for s in self[:9]))) if self.front else ''
        back = ','.join(map(str, (s for s in self[9:]))) if self.back else ''
        score = ''
        if not front:
            scores = ','.join(map(str, self)) + ' Incomplete'
        elif not back:
            scores = front
            score = f' ({self.front})'
        else:
            scores = ', '.join((front, back))
            score = f" : {self.front} + {self.back} = {self.score}"
        return f"[Day {self.name}: {scores}{score}]"

class Course(list):
    """Object for Storing All Data for a Particular Course, and Doing Probability Calculations/Analysis, and Making Plots"""
    def __init__(self, course:str = params.dataset):
        """Load the Data for the Specified Course ('course' must be in the 'courses' list)"""
        # Load the Data
        self.course = course
        self.index = params.courses.index(course)
        self.extend(map(Round, readlines(f"{course}.txt")))
        self.used = self[params.warmup_days:]
        self.current_day_front_half = self[-1] if self and len(self[-1]) == 9 else []
        self.data = np.array([r for r in self.used if r.complete], int)
        
        # Other Attributes
        self._cache = {} # cache for computing the probability tree
        self._hits = 0
        self._misses = 0
        
        # Pre-Processing
        self.days = len(self.data)
        self.total_days = len(self)
        self.scores = [r.score for r in self if r.complete]
        self.scores_by_half = ([r.front for r in self if r.completion&1], [r.back for r in self if r.completion&2])
        self.min_score = int(np.min(np.sum(self.data, axis = 1)))
        self.max_score = int(np.max(np.sum(self.data, axis = 1)))
        self.min_score_possible = int(np.sum(np.min(self.data, axis = 0)))
        self.max_score_possible = int(np.sum(np.max(self.data, axis = 0)))
        
        # Tally the Shots
        self.weights = None
        if not params.use_weights:
            # Weight all days the same
            self.counts = [Counter(self.data[:,i]) for i in range(18)]
            for d, s in enumerate(self.current_day_front_half):
                self.counts[d][s] += 1
            
            # Update Things for Specific Courses (e.g. Big Putts courses tend to reuse hole 1 as hole 18)
            if params.dataset == 'bigputts':
                self.counts[0] = Counter([r[0] for r in self.used] + [r[17] for i, r in enumerate(self) if i >= params.warmup_days and (i < 10 or i >= 24)])
                self.counts[-1] = Counter([r[17] for r in self.used[max(params.warmup_days, 10):24]]) if params.bp18 else self.counts[0].copy()
            elif params.dataset == 'bigputts waukesha':
                self.counts[0] = Counter([r[0] for r in self.used if len(r) > 0] + [r[17] for r in self.used if len(r) == 18])
                self.counts[-1] = self.counts[0].copy()
        else:
            # weight more recent days more
            self.counts = [defaultdict(float) for i in range(18)]
            self.weights = np.linspace(1 - params.weight_spread, 1 + params.weight_spread, len(self.used))
            for r, w in zip(self.used, self.weights):  #  round, weight
                for h, s in enumerate(r):  #  hole, score
                    self.counts[h][s] += w
            
            # Update Things for Specific Courses
            if params.dataset == 'bigputts':
                self.counts[0] = defaultdict(float)
                self.counts[17] = defaultdict(float)
                for i, (r, w) in zip(self.used, start = params.warmup_days):
                    self.counts[0][r[0]] += w  #  hole 1 always counts towards hole 1
                    if i < 10 or i >= 24: self.counts[0][r[17]] += w  #  hole 18 sometimes counts towards hole 1
                    elif params.bp18: self.counts[17][r[17]] += w  #  hole 18 sometimes counts towards itself
                if not params.bp18: self.counts[17] = self.counts[0].copy()  #  hole 18 = hole 1 based on the input parameters
            elif params.dataset == 'bigputts waukesha':
                # Combine Holes 1 and 18 Directly
                counts = defaultdict(float)
                for r, w in zip(self.used, self.weights):
                    counts[r[0]] += w
                    counts[r[17]] += w
                self.counts[0] = counts.copy()
                self.counts[17] = counts.copy()
        
        # Compute the Shot Probabilities
        sums = [sum(hole.values()) for hole in self.counts]
        self.shot_probabilities = [[hole[s] / n for s in range(1, max(hole.keys()) + 1)] for hole, n in zip(self.counts, sums)]
        
        # Factor In Historical Data
        if params.use_historical and params.dataset in ('gastraus'):
            # Get the Data
            if params.dataset == 'gastraus':
                N = [46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 50, 48, 50, 50, 46, 46]
                C = [2, 0, 1, 3, 1, 10, 0, 4, 0, 1, 2, 1, 2, 0, 0, 0, 4, 2]
            
            # Compute the Scramble Probabilities
            P = [c / n for n, c in zip(N, C)]
            P = [1 - (1 - p)**2 for p in P]
            
            # Update the Shot Probabilities
            alpha = 0.5
            if params.use_historical == 2:
                # Weight Only by Alpha
                for p, Q in zip(P, self.shot_probabilities):
                    Q[0] = (alpha * p + (1 - alpha) * Q[0])
                    k = (1 - Q[0])/sum(Q[1:])
                    for i in range(1, len(Q)):
                        Q[i] *= k
            else:
                # Weight by Alpha and by Number of Observations
                for n, p, Q in zip(N, P, self.shot_probabilities):
                    Q[0] = 2 * ((alpha * n * p + (1 - alpha) * len(self.data) * Q[0])) / (n + len(self.data))
                    k = (1 - Q[0])/sum(Q[1:])
                    for i in range(1, len(Q)):
                        Q[i] *= k
        
        # Assert a Minimum Hole-In-One Probability
        if params.min_HI1_probability > 0:
            for hole in self.shot_probabilities:
                if hole[0] < params.min_HI1_probability:
                    # Adjust the Hole-In-1 Probability
                    hole[0] = params.min_HI1_probability * 1 / (1 - params.min_HI1_probability*len(hole))
                    
                    # Normalize the Probability Distribution for the Hole
                    s = sum(hole)
                    for i in range(len(hole)):
                        hole[i] /= s
        
        # Compute the Expected Number of Shots
        self.expected_shots = [sum(p*t for t, p in enumerate(hole, start = 1)) for h, hole in enumerate(self.shot_probabilities, start = 1)]
    
    def __repr__(self) -> str:
        if params.warmup_days > 0 and len(self) >= params.warmup_days:
            n = 45 + len(self[params.warmup_days - 1].name)
            return f"{self.course} : [\n\t" + ',\n\t'.join(str(r) for r in self[:params.warmup_days]) + ',\n\t' + '-'*n + '\n\t' + ',\n\t'.join(str(r) for r in self[params.warmup_days:]) + '\n]'
        else:
            return f"{self.course} : [\n\t" + ',\n\t'.join(str(r) for r in self) + '\n]'
    
    def path_probability(self, path:list, start:int = 0) -> float:
        """Compute the Likelihood of a Particular Round"""
        return product((self.shot_probabilities[h][s - 1] for h, s in enumerate(path, start = start)))
    
    def estimate_single_player_probabilities(self, plot:bool = True):
        """Estimate the Shot Probabilities for a Danny/Steven Individually from their 2-Man Scramble Results"""
        def get_hole_probabilities(q_hole):
            P = []
            S = 0
            for q in reversed(q_hole):
                a, b = quadratic(1, 2*S, -q)
                p = a if 0 <= a <= 1 else b
                P.append(p)
                S += p
            return P[::-1]
        
        P_single = list(map(get_hole_probabilities, self.shot_probabilities))
        
        if plot:
            self.plot_hole_probabilities(P_single)
        
        return P_single
    
    """-------------------- ANALYSIS --------------------"""
    
    def P(self, hole:int = 0, shots:int = params.target, exact:bool = False, end = 18):
        """Estimate the Likelihood of Beating the Target by Trying All Possible Options
        
        THIS IS THE MAIN METHOD FOR MY ANALYSIS 
        
        It recursively searches the entire probability tree to figure out the probability they'll complete 
        all the holes starting with hole <hole> and ending with hole <end - 1> (holes are 0-indexed, 
        so they go from 0-17, not from 1-18), given <shots> total shots. If <exact> == True, it computes the 
        probability they'll use exactly all of the shots they've been given (i.g. "what is the probability they 
        shoot a 29"), and if it's false it computes the probability they'll use complete the holes with shots 
        remaining (i.e. "what is the probability they shoot a 29 or better").
        
        To compute probabilities for the back 9, use calls like Course.P(9, 29 - x), where x is their score on 
        the front 9. Similarly, to compute probabilities for the front 9, use calls like Course.P(shots = x, end = 9) 
        where x is the number of shots they're allowed on the front 9. By similar calls you can compute probabilities 
        for any arbitrary stretch of holes.
        
        A more "math" version of what it's computing is:
            P(score == <shots> over holes [<hole>, <end>))  if exact
            P(score <= <shots> over holes [<hole>, <end>))  else
        where holes are 0-indexed
        
        This function uses the recursion relation:
            Course.P(h, s) = P(1 on hole h) * Course.P(h, s - 1) + P(2 on hole h) * Course.P(h, s - 2)) + ...
        (which is kinda just how probability trees work lol)
        And the base case:
            Course.P(end, shots, True, end) = 1 if shots == 0 else 0
        (i.e. if you have no holes remaining and no shots remaining, then you've completed all the holes 
        using exactly the specified number of shots. And otherwise, you haven't)
        
        This method is cached to improve speed, and prunes where possible (i.e. if a computation is easily 
        known to result in 0 contribution to the final result it is skipped). The cache's hit rate isn't 
        that high when it first computes the tree, but without caching this computation takes a very long 
        time to complete (and with it it's almost instantaneous, running in under 1ms on my computer).
        """
        
        # Check the Cache, Reusing Computations Whenever Possible
        state = (hole, shots, exact, end)
        if state in self._cache:
            self._hits += 1
        else:
            self._misses += 1
            if shots < end - hole:
                # base case: only getting holes-in-one is no longer good enough, so the probability is 0
                self._cache[state] = 0
            elif not exact:
                # P(total_shots <= <shots>) = P(total_shots = 18) + P(total_shots = 19) + ... + P(total_shots = <shots>)
                # Doing it this way, rather than by editing the base case saves cache space and computation time by about a factor of 2
                # (which doesn't really matter, but still, why not lol)
                self._cache[state] = self.P(hole, shots - 1, False, end) + self.P(hole, shots, True, end)
            elif hole == end:
                # base case: there are no more holes to play -> return if they have no extra shots remaining
                self._cache[state] = int(shots == 0)
            else:
                # recursive case -> sum the contribution from getting a every possible score on the current hole
                self._cache[state] = float(sum(p * self.P(hole + 1, shots - s, exact, end) for s, p in enumerate(self.shot_probabilities[hole], start = 1) if p > 0))
        
        # Return the Memorized Result
        return self._cache[state]
    
    def P_success_given_front(self, front_score:int, total_shots:int = params.target, exact:bool = False, end:int = 18):
        """Compute the Probability of Completing the Challenge Given a Score on the Front 9"""
        return self.P(9, total_shots - front_score, exact, end)

    def P_success_given_partial(self, scores:list, total_shots:int = params.target, exact:bool = False, end:int = 18):
        """Compute the Probability of Completing the Challenge Given the First n Holes"""
        return self.P(len(scores), total_shots - sum(scores), exact, end)

    # Generate All Possible "Paths" to Beating the Challenge
    def Paths(self, shots:int = params.target, hole:int = 0, end = 18, exact:bool = False) -> list:
        """Generate All Possible "Paths" to Beating the Challenge
        
        This effectively does the same computation as Course.P(), except that it tracks the paths, and can't be cached. 
        This means it's significantly slower, so it is not used in the main analysis, but can be used to compute interesting 
        metrics like "what score to they realistically need on the front 9 to have good odds on the day". That being said, 
        those sorts of metrics can also be computed using Course.P() if you're clever.
        """
        
        # a recursive helper function which walks along all possible paths
        start = hole
        path = []
        def helper(hole:int, shots:int, P:float):
            if hole == end:
                if P > 0 and shots == 0 or (not exact and shots > 0):
                    yield (path.copy(), self.path_probability(path, start))
            elif P > 0 and shots >= end - hole:
                for s, p in enumerate(self.shot_probabilities[hole], start = 1):
                    path.append(s)
                    yield from helper(hole + 1, shots - s, P * p)
                    path.pop()
        
        # use the helper
        yield from helper(hole, shots, 1.0)

    def montecarlo(self, N:int = 1000, shots:int = params.target, gaussian:bool = True, output:int = 0):
        """Simulate N Rounds of Minigolf to Estimate the Rate of Shooting <shots> or Better 
        (either modeling each hole as a guassian, or by using 'shot_probabilities')
        
        In practice, Course.P() should always give more accurate results (and run faster); 
        and as a result, this is not used in my main analysis.
        """
        
        # Define a Sample Function
        warning = None
        if gaussian:
            # Get the (Weighted) Probability Distributions for Each Hole
            W = np.array([[w for _ in range(18)] for w in self.weights]) if type(self.weights) == np.ndarray else 1
            means = np.mean(self.data * W, 0)
            devs = np.std(self.data, 0)
            
            # Define the Sample Function
            def r_shots(hole:int):
                return max(int(round(normal(means[hole], devs[hole]))), 1)
        else:
            # Compute the Cumulative Probability Distribution for Each Hole
            cdfs = [np.cumsum(hole) for hole in self.shot_probabilities]
            
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
        z = (shots + (params.z_score_target - params.target) - mean) / dev
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

    def hole_in_one_analysis(self, print_results:bool = True):
        # Count the Holes in X
        counts = [Counter(np.sum(self.data == x, axis = 1)) for x in range(1, np.max(self.data) + 1)]
        for score, count in Counter(self.current_day_front_half).items():
            counts[score - 1][count] += 1
        
        # compute the probability of shooting X Y's (e.g. probability of shooting 7 1's)
        @cache
        def Q(count, score):
            return counts[score - 1][count] / sum(counts[score - 1].values())
        
        # a helper to find all the options
        @cache
        def P(i:int = 0, holes:int = 18, shots:int = params.target, exact:bool = False):
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
        s = sum(P(shots = score, exact = True) for score in range(self.min_score_possible, self.max_score_possible + 1))
        
        # Use the Recursive Solver to Compute the Expected Number of Shots
        e = sum(P(shots = score, exact = True) * score / s for score in range(self.min_score_possible, self.max_score_possible + 1))
        v = sum(P(shots = score, exact = True) * score**2 / s for score in range(self.min_score_possible, self.max_score_possible + 1))
        v = np.sqrt(v - e**2)
        
        # print or return the result
        if print_results:
            print(f'Probability = {p:.5f}')
            print(f'Expected Days: {1/p:.1f}')
            print(f'Expected Score: {e:.2f} ± {v:.2f}')
        else:
            return p, e, v

    def rank_holes(self, expectation_value:bool = False, print_results:bool = True, plot:bool = True):
        """Rank the Holes Based on how 'Good' they are (either by hole-in-1 probability, or by expected shots)"""
        expectation_value = int(bool(expectation_value))
        holes = [(h, P[0], e) for h, (P, e) in enumerate(zip(self.shot_probabilities, self.expected_shots), start = 1)]
        holes.sort(key = lambda x : (x[1], -x[2])[::(1 - 2*expectation_value)], reverse = True)
        
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
            print('Hole Ranking: Hole-In-One Probability (Expected Shots)')
            for h, p, e in holes:
                print(f"{h}\t:\t{100*p:.1f}%\t({e:.2f})")
        else:
            return holes

    def plot_round_probability(self, scores:list = 0, starting_hole:int = 1):
        """Plot the Probability of Success Before Each Shot Throughout a Round (<scores> can also be the 1-indexed day number: 0 -> most recent round)"""
        if type(scores) == int:
            scores = self[scores - 1 if scores >= 0 else scores]
        
        # Get the Probabilities
        X = []
        P = []
        S = 0
        for h, s in enumerate(scores, start = starting_hole):
            X.append(h)
            P.append(self.P(h - 1, params.target - S))
            S += s
        
        # Plot the Result
        fix, ax = plt.subplots()
        plt.errorbar(X, P, fmt = '.', color = 'violet')
        plt.plot(X, P, color = 'violet')
        plt.ylabel(f'Probability of Breaking {params.target + 1}')
        plt.xlabel('Hole')
        plt.xticks(X, X)
        plt.show()

    def plot_expected_shots(self, cumulative:bool = False):
        """Either Plots the Expected Number of Shots for Each Hole, 
        or the Expected Number of Total Shots After Each Hole"""
        fig, ax = plt.subplots()
        x = list(range(1, 19))
        if cumulative:
            y = np.cumsum(self.expected_shots)
            plt.plot(x, y, color = 'black')
            plt.ylabel('Total Shots')
        else:
            y = self.expected_shots
            plt.errorbar(x, y, fmt = '_', color = 'black')
            plt.ylabel('Shots')
        plt.xlabel('Hole')
        plt.xticks(x, x)
        plt.ylim(1, round(2 * max(y) + 1) / 2)
        plt.show()

    def plot_halves(self, return_data:bool = False, print_results:bool = True):
        """Plots the Likelihood of Various Scores on Either the Front 9 or the Back 9"""
        
        # Compute the Max Score for Either Half
        endpoints = [(0, 9), (9, 18)]
        max_score = max(sum(map(len, self.shot_probabilities[start:end])) for start, end in endpoints)
        
        # Get the Data
        x = list(range(9, max_score + 2))
        P = [[self.P(start, s, True, end) for s in x] for start, end in endpoints]
        
        # Plot the Results
        fig, ax = plt.subplots()
        for i, (label, color) in enumerate(zip(['Front 9', 'Back 9'], ['violet', 'teal'])):
            plt.plot(x, P[i], color = color, label = label)
        plt.xlabel("Score")
        plt.ylabel('Likelihood')
        plt.legend()
        plt.show()
        
        # Calculate and Print Some Stats
        means = [np.mean(half) for half in self.scores_by_half]
        devs = [np.std(half) for half in self.scores_by_half]
        s = 0
        u = 0
        for half, mean, dev in zip(['Front', 'Back'], means, devs):
            if print_results: print(f"{half} Half: {mean:.2f} ± {dev:.2f} shots")
            s += mean
            u += dev**2
        u = np.sqrt(u)
        z = (params.z_score_target - s) / u
        p = norm.sf(-z)
        if print_results: print(f"Overall: {s:.2f} ± {u:.2f} shots -> z = {z:.3f} -> p = {p:.5f} -> Exp Days = {1/p:.1f}")
        
        # Return the Data
        if return_data:
            return P, u, s, z, p

    def analyze_front_vs_back(self):
        """Generates a Plot Showing What Score Needs to be Achieved on the Front 9
        
        This multiplies the probability of scoring exactly S on the front half by 
        the probability of beating the target (29) given a score S on the front half, 
        and plots the results for all possible scores on the front half.
        """
        
        # Get the Data
        x = list(range(9, params.target - 9))
        probabilities = [self.P(0, s, True, 9) * self.P(9, params.target - s, False, 18) for s in x]
        
        # Plot the Result
        fig, ax = plt.subplots()
        plt.plot(x, probabilities, color = 'violet')
        plt.ylabel('Overall Likelihood of Success')
        plt.xlabel('Score on the Front 9')
        plt.show()
        
        # Return the Data
        return probabilities

    def plot_P_success_given_front(self, front_scores = range(9, params.target - 9)):
        """Generate a Plot Showing the Probability of Completing the Challenge Given a Score on the Front 9"""
        # Get the Data
        P = [self.P(9, params.target - s, False, 18) for s in front_scores]
        
        # Plot the Result
        fig, ax = plt.subplots()
        plt.plot(front_scores, P, color = 'violet')
        plt.ylabel('Likelihood of Success')
        plt.xlabel('Score on the Front 9')
        plt.show()
        
        # Return the Data
        return P

    def plot_score_distribution(self, shots:int = 0, start:int = 0, end:int = 18, include_full_round:bool = True, include_histogram:bool = False, include_cumulative:bool = False):
        """Plot the Expected Distribution of Scores Over the Run of Holes Specified"""
        # Update the include_full_round Flag
        include_full_round = include_full_round and (shots > 0 or start > 0) and (not include_histogram or 2 in (include_full_round, include_histogram))
        
        # Compute a List of All Possible Numbers of Shots Remaining, Then Compute the Probability for Each
        S = list(range(shots + end - start, shots + 2 + sum(map(len, self.shot_probabilities[start:]))))
        P = [self.P(start, s - shots, True, end) for s in S]
        P2 = [self.P(0, s, True, end) for s in S] if include_full_round else []
        
        # Get the Histogram Data
        counts = defaultdict(int, Counter(int(shots + s) for s in np.sum(self.data[:,start:end], 1)))
        rounds = sum(counts.values())
        for score in counts:
            counts[score] /= rounds
        
        # Get the Cumulative Data
        C = [0]*len(P)
        for i, p in enumerate(P): C[i] = C[i - 1] + p
        C2 = [0]*len(P)
        for i, p in enumerate(P2): C2[i] = C2[i - 1] + p
        
        # Compute the Expected Number of Shots
        #E = sum(s*p for s, p in zip(S, P))
        
        # Plot the Result
        fix, ax = plt.subplots()
        #plt.plot([E, E], [0, max(P)], '--', color = 'lightgrey', label = f'Expected Shots = {E:.2f}')
        plt.plot(S, P, color = 'violet', label = f'After the First {start} Holes' if start else 'P(score == x)', zorder = 100)
        if include_full_round: plt.plot(S, P2, '--', color = 'lightgrey', label = 'Before the First Hole', zorder = 99)
        if include_histogram: plt.bar(S, [counts[s] for s in S], color = 'lightpink', alpha = 0.7, zorder = 96)
        if include_cumulative:
            plt.plot(S, C, '-.', color = 'teal', label = 'P(score <= x)', zorder = 98)
            if include_full_round: plt.plot(S, C2, '-.', color = 'lightgrey', zorder = 97)
        plt.xlabel('Total Shots' + f'\n(sitting on {shots} after {start} holes)'*bool(shots or start))
        plt.ylabel('Likelihood' if include_cumulative else 'Likelihood : P(score == x)')
        if include_full_round or include_cumulative: plt.legend()
        plt.show()
        
        # Return the Data
        return P
    
    def plot_hole_probabilities(self, probabilities:list = None, ranked:bool = True, colors = ['violet', 'mediumorchid', 'purple', 'indigo', 'maroon', 'red'], width = 0.75):
        """Plot the Hole Probabilities as a Stacked Bar Plot"""
        
        # get the data in the proper format
        probabilities = self.shot_probabilities if probabilities == None else probabilities
        y_bins = max(map(len, probabilities))
        if ranked:
            ranked = [(h, probabilities[h - 1], e) for h, p, e in self.rank_holes(ranked == 2, False, False)]
            x = [x for x, _, _ in ranked]
            e = [e for _, _, e in ranked]
            probabilities = [np.array([hole[s] if s < len(hole) else 0 for _, hole, _ in ranked]) for s in range(y_bins)]
        else:
            x = list(range(1, 19))
            e = self.expected_shots.copy()
            probabilities = [np.array([hole[s] if s < len(hole) else 0 for hole in probabilities]) for s in range(y_bins)]
        
        # Make the Main Figure
        xx = list(range(1, 19))
        bottom = np.zeros(18)
        fig, ax = plt.subplots()
        for s in range(y_bins):
            ax.bar(xx, probabilities[s], width, label = str(s + 1), bottom = bottom, color = colors[s])
            bottom += probabilities[s]
        
        # Add the Text/Legend
        x0, y0, w, h = ax.get_position().bounds
        for X, E in zip(xx, e): fig.text(x0 + (X + 0.05)*w/19.5, 0.9, f"{E:.2f}", fontsize = 8, rotation = 90, rotation_mode = 'default', transform_rotates_text = True)
        ax.set_xlabel('Hole')
        ax.set_ylabel('Probability')
        ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        plt.xticks(xx, x)
        
        # Show the Figure
        plt.show()
        
    def all_plots(self, score_histogram:bool = False, round_shots:list = 0):
        self.plot_score_distribution(include_histogram = score_histogram, include_cumulative = True)
        # rank_holes() # hole ranking happens in self.plot_hole_probabilities()
        # plot_expected_shots() # expected shots are included in self.plot_hole_probabilities()
        self.plot_halves(print_results = False)
        self.analyze_front_vs_back()
        self.plot_P_success_given_front()
        self.plot_hole_probabilities()
        if self.current_day_front_half: self.plot_score_distribution(sum(self.current_day_front_half), 9, 18)
        self.plot_round_probability(round_shots)

    def analyze(self):
        # Compute the Round Score Distribution Using the Recursive Solver
        X = np.array(range(3*18 + 1))
        exact_probabilities = np.array([self.P(shots = s, exact = True) for s in X])
        probabilities = np.array([self.P(shots = s) for s in X])
        expected_score = sum(s*p for hole in self.shot_probabilities for s, p in enumerate(hole, start = 1))
        expected_var = sum(sum(s*s*p for s, p in enumerate(hole, start = 1)) - sum(s*p for s, p in enumerate(hole, start = 1))**2 for hole in self.shot_probabilities)
        expected_stdev = np.sqrt(expected_var)
        z0 = (params.z_score_target - expected_score) / expected_stdev
        p0 = norm.sf(-z0)
        
        # Do Gaussian Fits
        parameters, var = curve_fit(gaussian, np.array(range(0, len(exact_probabilities))), exact_probabilities)
        var = np.sqrt(np.diag(var))
        mean_shots = np.mean(self.scores * (self.weights if type(self.weights) == np.ndarray else 1))
        std_shots = np.std(self.scores)
        
        # Mean/Dev of Scores by Half Analysis
        e5 = sum(np.mean(half) for half in self.scores_by_half)
        u5 = np.sqrt(sum(np.std(half)**2 for half in self.scores_by_half))
        z5 = (params.z_score_target - e5) / u5
        p5 = norm.sf(-z5)
        
        # Do a Hole-In-1 Count Analysis
        p4, s4, v4 = self.hole_in_one_analysis(False)
        
        # Run a Quick Monte-Carlo Sim (using a Gaussian Fit for Each Hole)
        mean, dev, p3 = self.montecarlo(N = 1000, gaussian = True, output = 2)
        
        # Compute the Z-Scores and P Values for the Fits (assuming anything less than 29.5 "breaks 30")
        z1 = (params.z_score_target - mean_shots) / std_shots
        z2 = (params.z_score_target - parameters[0]) / parameters[1]
        p1 = norm.sf(-z1)
        p2 = norm.sf(-z2)
        
        # Get the Expected Number of Days From a P Value
        def E(p:float) -> str:
            return 'NaN  ' if p <= 0 else f"{1 / p + params.warmup_days:.2f}"
        
        # Print the Results
        print('\n'.join([
            'Method\t\t\tMean\t\tSTDev\t\tZ-Score\t\tProbability\t\tExpected Days', 
            '-'*81,
            f'Tree Search\t\t{expected_score:.2f}\t\t \t\t\t\t\t\t  {probabilities[params.target]:.5f}\t\t\t{E(probabilities[params.target])}',
            '-'*81,
            f'Tree Search*\t{expected_score:.2f}\t\t{expected_stdev:.3f}\t\t{z0:.3f}\t\t  {p0:.5f}\t\t\t{E(p0)}',
            f'Gaussian Fit*\t{parameters[0]:.2f}\t\t{parameters[1]:.3f}\t\t{z2:.3f}\t\t  {p2:.5f}\t\t\t{E(p2)}',
            f'Round Scores\t{mean_shots:.2f}\t\t{std_shots:.3f}\t\t{z1:.3f}\t\t  {p1:.5f}\t\t\t{E(p1)}',
            f'Half Scores\t\t{e5:.2f}\t\t{u5:.3f}\t\t{z5:.3f}\t\t  {p5:.5f}\t\t\t{E(p5)}',
            #f'HIO Analysis\t{s4:.2f}\t\t{v4:.3f}\t\t\t\t\t  {p4:.5f}\t\t\t{E(p4)}',
            f'Monte Carlo\t \t{mean:.2f}\t\t{dev:.3f}\t\t\t\t\t  {p3:.5f}\t\t\t{E(p3)}',
            '*based on Tree Search results'
        ]))
        
        # Print the Expected Shots
        f = sum(self.expected_shots[:9])
        b = sum(self.expected_shots[9:])
        t = sum(self.expected_shots)
        easier = 'Front' if f < b else 'Back'
        print(f'\nExpected Shots: {f:.2f} (Front) + {b:.2f} (Back) = {t:.2f} (Total)')
        print(f'{easier} is {abs(f - b):.2f} shots easier')
        
        # Print the Odds of Completing the Challenge on the Current Dat
        if self.current_day_front_half:
            score = sum(self.current_day_front_half)
            p = self.P(9, params.target - score)
            e = score + b
            s = (sum(self.P(9, s, True, 18) * (s + score)**2 for s in range(9, sum(map(len, self.shot_probabilities[9:])) + 1)) - e**2)**0.5
            z = (params.z_score_target - e) / s
            p2 = norm.sf(-z)
            print(f'\nProbability of Winning Today (Sitting on {score}): {100*p:.2f}%\nExpected Shots = {e:.3f} ± {s:.3f} (z = {z:.3f}, p = {100*p2:.3f}%)')
        else:
            p = self.path_probability(self.data[-1])
            pr = p / product((max(hole) for hole in self.shot_probabilities))
            print(f'\nRelative Likelihood of Previous Round: {pr:.3f} (absolute = {100*p:.3f}%)')
        
        if False:
            r = np.corrcoef(list(range(len(self.scores))), self.scores)
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
        
        # Make the Plots
        self.all_plots()
        
        # Return the Data and the Plot Function
        return probabilities, exact_probabilities

"""-------------------- Run the Main Analysis --------------------"""

# LOAD THE DATA
course = Course()
shot_probabilities = course.shot_probabilities
expected_shots = course.expected_shots
course.analyze()
