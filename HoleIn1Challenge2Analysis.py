# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:00:50 2025

@author: Violet
"""

from HoleIn1ChallengeAnalysis import Course, params, readlines, np, plt, product, Counter
params.hio1p = True
params.alpha = 0.5  #  how much to weight the new data vs the old data
params.dataset = params.courses[0]  #  bigputts : 2-man scramble data to use
params.filepath = 'bigputts single-player.txt'  #  the path to the single-player data

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
            return f"{params.filepath.split('.')[0]} ({name}) : [\n\t{'\n\t'.join(map(str, self))}\n]"
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
        if state not in self._cache:
            if ones > stop - start:
                # Pruning/Base Case (there aren't enough holes to get the specified number of ones)
                self._cache[state] = 0
            elif not exact:
                # Non-Exact = Cumulative
                self._cache[state] = self.P(ones, start, stop, True) + self.P(ones + 1, start, stop, False)
            elif start == stop:
                # Base Case (there are no more holes left to play)
                self._cache[state] = int(ones == 0)
            elif ones:
                # Main Recursive Case (probability you get a one this hole, plus the probability you don't)
                p = self.probabilities[start]
                self._cache[state] = p * self.P(ones - 1, start + 1, stop, True) + (1 - p) * self.P(ones, start + 1, stop, True)
            else:
                # Secondary Recursive Case (ones are no longer allowed)
                #self._cache[state] = (1 - self.probabilities[start]) * self.P(ones, start + 1, stop, True, False)
                self._cache[state] = product(1 - p for p in self.probabilities[start:stop])
        
        # Return the Cached Result
        if return_uncertainty:
            return self._cache[state], self._cache[state] * sum(self.variances[start:stop])**0.5
        else:
            return self._cache[state]
    
    def P2(self, ones:int = 7, start:int = 0, return_uncertainty:bool = True):
        """Compute the Probability of Scoring Exactly the Specified Number of Ones in a Round (with early stopping)"""
        return self.P(ones, start, 18 - max(7 - ones), True, return_uncertainty)
    
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
    
    def analyze(self):
        pass
    
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
        p = [self.P(ones, start, stop - early_stopping * max(7 - ones, 0), True, True) for ones in x] # stop is variable because the challenge stops early
        y = [p for p, _ in p]
        c = [0]*len(p)
        c[-1] = p[-1][0]
        for i in range(len(p) - 2, -1, -1): c[i] = p[i][0] + c[i + 1]
        counts = Counter(day.ones for day in self)
        bars = [counts[x] / len(self) for x in x]
        
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
    
    def plot_expected_ones(self, target:int = 7, exact:bool = False):
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
        #plt.fill_between(X, E - uE, E + uE, color = 'violet', alpha = 0.2)
        plt.xticks(ticks)
        plt.xlabel('Hole')
        plt.ylabel('Expected Ones')
        plt.show()

# Load the Data
danny = Player('d')
steven = Player('s')