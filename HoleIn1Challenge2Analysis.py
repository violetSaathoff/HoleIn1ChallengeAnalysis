# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:00:50 2025

@author: Violet
"""

from HoleIn1ChallengeAnalysis import Course, params, readlines, np, plt, product, Counter, norm
params.hio1p = True
params.new_putters = True
params.alpha = 1  #  how much to weight the new data vs the old data
params.dataset = params.courses[0]  #  bigputts : 2-man scramble data to use
params.filepath = 'bigputts single-player.txt'  #  the path to the single-player data
params.hole18 = [0]  # hole 18 = hole 1

class Day(list):
    def __init__(self, line:str):
        day, line = line.split(':')
        line = line.replace(' ', '')
        self._day = day.rstrip(' ')
        self.day = int(''.join(c for c in day if c.isdigit()))
        self.name = ''.join(c for c in day if not c.isdigit() and c != ' ')
        self.extend(map(int, line))
        self.holes = len(self)
        self.ones = self.count(1)
        self.rate = self.ones / self.holes
        self.front = self[:9].count(1) if len(self) >= 9 else None
        if self._day.endswith(self.name): self._day = self._day[:-len(self.name)]
    
    def __repr__(self):
        front = ''.join(map(str, self[:9]))
        back = ''.join(map(str, self[9:]))
        front = front + '.'*(9 - len(front))
        back = back + '.'*(9 - len(back))
        return f"[{self._day} : {front} {back} ({self.ones})]"

class Player(list):
    def __init__(self, name:str):
        # Load the Data
        self.name = name
        days = [Day(l) for l in readlines(params.filepath)]
        if not params.new_putters: days = [day for day in days if 39 <= day.day < 49]
        self.extend(day for day in days if day.name == self.name)
        
        # Compute the Hole Probabilities
        self.weights = None
        self.counts = None
        self.probabilities = None
        self.variances = None
        self.variance = None
        self.compute_probabilities()
        self._cache = {}
    
    def __repr__(self):
        name = 'steven' if self.name == 's' else 'danny'
        if len(self):
            days = '\n\t'.join(map(str, self))
            return f"{params.filepath.split('.')[0]} ({name}) : [\n\t{days}\n]"
        else:
            return f"[{params.filepath.split('.')[0]} ({name}) : ]"
    
    def compute_probabilities(self):
        """Compute the Hole-In-One Probability for Each Hole (using the settings in the static class <params>)"""
        # Count the 1s
        self.counts = [[0, 0] for _ in range(18)]
        self.weights = list(map(float, np.linspace(1 - params.weight_spread, 1 + params.weight_spread, len(self)))) if params.use_weights else [1]*len(self)
        for day, weight in zip(self, self.weights):
            for hole, score in enumerate(day):
                self.counts[hole][1 - score] += weight
        
        # Compute the Probabilities
        self.probabilities = [float(ones / (ones + others)) if (ones + others) else 0 for ones, others in self.counts]
        
        # Edit the Hole 18 Probability
        if params.hole18 != [17]: 
            totals = [sum(c) for c in self.counts]
            self.probabilities[17] = sum(self.probabilities[i] * totals[i] for i in params.hole18) / sum(totals[i] for i in params.hole18)
            self.counts[-1] = [sum(self.counts[i][j] for i in params.hole18) for j in range(2)]
        
        # Incorporate Counts from the Original 2-Man Scramble Challenge
        if params.alpha != 1:
            course = Course()
            for h, (p, q) in enumerate(zip(self.probabilities, course.shot_probabilities)):
                q = q[0]
                c1 = sum(self.counts[h])
                c2 = sum(course.counts[h].values())
                self.counts[h][0] = float(params.alpha * p * c1 + (1 - params.alpha) * q * c2)
                self.counts[h][1] = float(params.alpha * (1 - p) * c1 + (1 - params.alpha) * (1 - q) * c2)
                self.probabilities[h] = float(self.counts[h][0] / sum(self.counts[h]))
                #self.probabilities[h] = float((params.alpha * p * c1 + (1 - params.alpha) * q * c2) / (params.alpha * c1 + (1 - params.alpha) * c2))
        
        # Estimate the Variance on Everything
        self.variances = [float(p * (1 - p) / sum(counts)) if sum(counts) else 1/len(self) for p, counts in zip(self.probabilities, self.counts)]
        if params.hole18 != [17]: self.variances[17] = sum(self.variances[h] for h in params.hole18) / len(params.hole18)
        self.variance = sum(self.variances)
    
    def _compute_challenge_probabilities(self) -> list:
        # Count the 1's on Each Hole
        counts = [[0, 0] for _ in range(18)]
        for day, weight in zip(self, self.weights):
            for hole, score in enumerate(day):
                counts[hole][1 - score] += weight
        
        # Compute and Return the Probabilities
        return [float(ones / (ones + others)) for ones, others in counts if (ones + others)]
    
    def P(self, ones:int = 7, start:int = 0, stop:int = 18, exact:int = False, return_uncertainty:bool = False):
        """Compute the Probability of Getting <ones> Holes-in-One on Holes <start> to <stop>"""
        # Check the Cache
        state = (ones, start, stop, exact)
        if return_uncertainty:
            p = self.P(ones, start, stop, exact, False)
            return p, p * sum(self.variances[start:stop])**0.5
        elif ones > stop - start or stop < start or ones < 0:
            # Pruning/Base Case (there aren't enough holes to get the specified number of ones)
            # Bad Case: the starting hole is after the ending hole
            # Bad Case: the target number of ones is negative
            return 0
        elif state in self._cache:
            pass #  don't re-compute non-trivial computations
        elif not exact:
            # Non-Exact = Cumulative
            self._cache[state] = self.P(ones, start, stop, True) + self.P(ones + 1, start, stop, False)
        elif start == stop:
            # Base Case: there are nor more holes to play, was the appropriate number of ones achieved?
            self._cache[state] = int(ones == 0)
        elif ones:
            # Main Recursive Case (probability you get a one this hole, plus the probability you don't)
            p = self.probabilities[start]
            self._cache[state] = p * self.P(ones - 1, start + 1, stop, True) + (1 - p) * self.P(ones, start + 1, stop, True)
        else:
            # Secondary Recursive Case (ones are no longer allowed)
            self._cache[state] = (1 - self.probabilities[start]) * self.P(ones, start + 1, stop, True, False)
            #self._cache[state] = product(1 - p for p in self.probabilities[start:stop])
        
        # Return the Cached Result
        return self._cache[state]
    
    def P2(self, ones:int = 7, start:int = 0, return_uncertainty:bool = True):
        """Compute the Probability of Scoring Exactly the Specified Number of Ones in a Round (with early stopping)"""
        return self.P(ones, start, 18 - max(6 - ones, 0), True, return_uncertainty)
    
    def E(self, ones:int = 7, exact:bool = False, start:int = 0, stop:int = 18, return_uncertainty:bool = True):
        """Compute the Expected Number of Days to Complete the Challenge (includes uncertainty propagation)"""
        p = self.P(ones, start, stop, exact, return_uncertainty)
        if return_uncertainty:
            p, u = p
            e = 1/p
            ue = u * e**2
            return e, ue
        else:
            return 1/p
    
    def ExpectedOnes(self, start:int = 0, stop:int = 18) -> (float, float, float):
        """Compute the Expected Number of Ones over the Specified Holes, as Well as both the Variance and the Uncertainty"""
        X = list(range(stop - start + 1))
        E = sum(self.P(ones, start, stop, True, False) * ones for ones in X)
        E2 = sum(self.P(ones, start, stop, True, False) * ones**2 for ones in X)
        S = abs(E**2 - E2)**0.5
        U = S / len(self)**0.5
        return E, S, U
    
    def analyze(self, ones:int = 7):
        """Estimate the Probability of Completing the Challenge, and the Number of Days to Complete the Challenge"""
        # Helper Function for Computing things via Z Scores
        def P(u:float, s:float):
            # Compute the Z Score
            z = (u - ones + 0.5) / s
            
            # Compute the Probability
            p = float(norm.sf(-z))
            
            # Estimate the Uncertainty on the Probability
            uz = 1 / len(self)**0.5  #  uncertainty on the z score
            up = uz * float(np.exp(-0.5 * z**2) / np.sqrt(8)) #  calculus approzimation for single-variable error propagation
            
            # Estimate the Number of Days
            e = 1/p
            ue = up * e**2
            
            # Return the Results
            return z, p, up, e, ue
        
        # Compute the Main Results
        p, up = self.P(ones, 0, 18, False, True)
        e, ue = self.E(ones, False, 0, 18, True)
        s, ss, _ = self.ExpectedOnes(0, 18)
        
        # Compute an Estimate of the Probability Using Z-Score Methods
        s2, ss2, = s, ss
        z2, p2, up2, e2, ue2 = P(s2, ss2)
        
        # Get the Expected Number of Days From a P Value
        def E(p:float, up:float = None) -> str:
            if p <= 0:
                return 'NaN' if up == None else 'NaN ± NaN'
            elif up == None:
                return f"{1 / p + params.warmup_days:.2f}"
            else:
                e = 1/p
                ue = up * e**2
                return f"{e + params.warmup_days:.2f} ± {ue:.2f}"
        
        # Print the Results
        def println(method:str, e_shots = None, s_shots = None, z = None, p = None, up = None, e_days = None, u_days = None):
            s_shots = '\t' if s_shots == None else f'{s_shots:.3f}'
            z = ' \t' if z == None else f'{z:.3f}'
            e = E(p, up) if e_days == None else f'{e_days:.2f} ± {u_days:.2f}'
            print(f'{method}\t{e_shots:.2f}\t\t{s_shots}\t\t{z}\t\t  {p:.5f}\t\t\t{e}')
        
        print('Method\t\t\tMean\t\tSTDev\t\tZ-Score\t\tProbability\t\t\tExpected Days')
        print('-'*89)
        println('Tree Search\t', s, ss, None, p, up, e, ue)
        #print('-'*89)
        println('Z-Score\t\t', s2, ss2, z2, p2, up2, e2, ue2)
        print('-'*89)
    
    def rank_holes(self, indices:bool = False, probabilities:list = None) -> list:
        if probabilities == None: probabilities = self.probabilities
        return sorted(range(1 - int(indices), len(probabilities) + 1 - int(indices)), key = lambda h : probabilities[h - 1 + int(indices)], reverse = True)
    
    def plot_probabilities(self, probabilities:list = None, ranked:bool = True, width = 0.75):
        """Plot the Hole Probabilities as a Stacked Bar Plot (probabilities = 'challenge' ignores historical data)"""
        
        # get the data in the proper format
        probabilities = self.probabilities if probabilities == None else probabilities
        probabilities = self._compute_challenge_probabilities() if probabilities == 'challenge' else probabilities
        holes = len(probabilities)
        x = self.rank_holes(False, probabilities) if ranked else list(range(1, holes + 1))
        y = [probabilities[h - 1] for h in x]
        probabilities = np.array([[p, 1 - p] for p in y]).transpose()
        
        # Make the Main Figure
        xx = np.array(range(1, holes + 1))
        bottom = np.zeros(holes)
        fig, ax = plt.subplots()
        colors = ['violet', 'mediumorchid', 'purple', 'indigo', 'maroon', 'red']
        for s in range(2):
            ax.bar(xx, probabilities[s], width, label = 'P(1)' if s == 0 else 'P(2+)', bottom = bottom, color = colors[s])
            bottom += probabilities[s]
        
        # Add the Text/Legend
        x0, y0, w, h = ax.get_position().bounds
        ax.set_xlabel('Hole')
        ax.set_ylabel('Probability')
        ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        plt.xticks(xx, x)
        
        # Show the Figure
        plt.show()
    
    def plot_distribution(self, start:int = 0, stop:int = 18, cumulative:bool = False, histogram:bool = False, early_stopping:bool = True):
        """Plot the Distirbution for the Total Number of Ones over the Specified Holes"""
        # Update the <early_stopping> Parameter
        if stop != 18: early_stopping = False
        
        # Get/Copmpute all the Data
        max_ones = sum(p > 0 for p in self.probabilities[start:stop])
        x = list(range(min(max_ones + 2, 19)))
        if early_stopping:
            p = [self.P2(ones, start, True) for ones in x]
        else:
            p = [self.P(ones, start, stop, True, True) for ones in x]
        y = [p for p, _ in p]
        c = [self.P(ones, start, stop - early_stopping * max(6 - ones, 0), False) for ones in x]
        counts = Counter(day.ones for day in self)
        bars = [counts[x] / len(self) for x in x]
        
        # Check that the Probabilities Sum to 1
        if abs(sum(y) - 1) > 1e-10:
            print(f'warning: probabilities sum to {sum(y)}')
        
        # Plot the Specified Results with Appropriate Lables/Formatting
        fig, ax = plt.subplots(1, 1)
        legend = False
        if histogram:
            plt.bar(x, bars, color = 'lightpink', alpha = 0.7, label = 'Actual Scores')
            legend = True
        plt.plot(x, y, color = 'violet', label = 'P(ones == x)')
        if cumulative:
            plt.plot(x, c, '-.', color = 'teal', label = 'P(ones >= x)')
            plt.ylabel('Probability')
            legend = True
        elif not histogram:
            y1 = [p - up for p, up in p]
            y2 = [p + up for p, up in p]
            plt.fill_between(x, y1, y2, color = 'violet', alpha = 0.2)
            plt.ylabel('P(ones == x)')
        if legend: plt.legend()
        plt.xlabel('Holes in One')
        plt.xticks(x)
        plt.show()
    
    def plot_expected_ones(self, target:int = 7, exact:bool = False, stdev:bool = False):
        """Plot the Expected Number of Ones after Each Hole for a Successful Run"""
        # Compute the Expected Number of Ones After the Specified Hole
        def Y(hole:int):
            X = list(range(hole + 1))
            P = [self.P(x, 0, hole, True, False) * self.P(target - x, hole, 18, exact, False) for x in X]
            EX = sum(x * p for x, p in zip(X, P)) / sum(P)
            sEX = abs(sum(x * p**2 for x, p in zip(X, P)) - EX**2)**0.5
            return EX, sEX
        
        # Compute the Expected Ones
        E = [Y(h) for h in range(19)]
        uE = np.array([ue for _, ue in E])
        E = np.array([e for e, _ in E])
        
        # Plot the Result
        X = [x + 0.5 for x in range(19)]
        ticks = list(range(1, 19))
        plt.plot(X, [target]*len(X), '--', color = 'lightgrey')
        plt.plot(X, E, color = 'violet')
        if stdev: plt.fill_between(X, E - uE, E + uE, color = 'violet', alpha = 0.2)
        plt.xticks(ticks)
        plt.xlabel('Hole')
        plt.ylabel('Expected Ones')
        plt.show()

# Load the Data
danny = Player('d')
steven = Player('s')

# Run the Main Analysis
print('Danny:')
danny.analyze()
print('\nSteven:')
steven.analyze()