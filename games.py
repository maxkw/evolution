from utils import flip
from itertools import product
from numpy import array
from copy import copy
import numpy as np
class Decision(object):
    """
    A decision is defined by a payoffDict
    one can play out a decision by feeding it agents and observers
    the agents will be permuted according tothe given order

    Tested
    """
    def __init__(self,payoffDict):
        actions =  payoffDict.keys()
        self.N_players = len(payoffDict.values()[0])
        self.actions = actions
        self.action_lookup = dict(map(reversed,enumerate(actions)))
        self.payoffs = payoffDict
        
    def __call__(self,action):
        self.last_action = action
        return self.payoffs[action]
    
    def play(decision,participants,observers=[],tremble=0):
        participant_ids = [participant.world_id for participant in participants]
        decider = participants[0]

        intention = decider.decide(decision,participant_ids)
        
        #if flip(tremble):
        #    action = np.random.choice(decision.actions)
        #else:
        action = intention

        payoffs = copy(decision(action))
            
        observer_ids = participant_ids#tuple(observer.world_id for observer in set(observers+participants))
        observations = [(decision,participant_ids,observer_ids,action)]

        return payoffs, observations

class DecisionObserved(Decision):
    """
    in this kind of decision, every observer observes after a decision is made
    """
    def play(self,participants,observers=[],tremble=0):
        payoffs, observations = super(DecisionObserved,self).play(decider,participants,observers,tremble)
        for observer in set(observers+participants):
            observer.observe_k(observations,observer.genome['K'],tremble)
        return payoffs, observations
    
class DecisionSeq(object):
    """
    A DecisionSequence is specified by:
    A sequence of Decision/order pairs.
    The length of an ordering must be the same as the number of players for the corresponding game

    NOTE:
    ALL GAMES ARE ASSUMED TO HAVE THE SAME NUMBER OF PLAYERS FOR NOW
    
    Tested
    """
    def __init__(self, decision_ordering_pairs):
        self.decision_ordering_pairs = decision_ordering_pairs
        
    def matchups(self,participants):
        return self.decision_ordering_pairs
    
    def play(self,participants,observers=[],tremble=0):
        observations = []
        extend_obs = observations.extend
        payoffs = np.zeros(len(participants))
        
        for decision,ordering in self.matchups(participants):
            pay,obs = decision.play(participants[ordering],observers,tremble)
            payoffs[ordering] += pay
            extend_obs(obs)

        return payoffs,observations

class DecisionSeqObserved(DecisionSeq):
    """
    Same as DecisionSequence except that after every sequence is done, every observer observes.

    Tested
    """
    def play(self,participants,observers=[],tremble=0):
        payoffs, observations = super(DecisionSeqObserved,self).play(participants,observers,tremble=0)
        for observer in set(observers+list(participants)):
            observer.observe_k(observations,observer.genome['RA_K'],tremble)
        return payoffs,observations
            
class SymmetricGame(DecisionSeqObserved):
    """
    plays the game with every meaningful permutation of deciders and non-deciders
    this game assumes that the order of non-deciders does not matter

    Tested
    """
    def __init__(self, payoffDict):
        N = len(payoffDict.values()[0])
        orderings = (range(n,n+1)+range(n)+range(n+1,N) for n in xrange(N))
        decisions = [(Decision(payoffDict),ordering) for ordering in orderings]
        self.N_players = N
        super(SymmetricGame,self).__init__(decisions)

class DecisionDependent(DecisionObserved):
    """
    This type of decision is defined solely by a payoff
    """
    def __init__(self,payoff):
        pass
    
class DecisionDependentSeq(DecisionSeq):
    """
    This Decision Seq is made up of observed decisions
    The first must be of type DecisionObserved
    The rest must be of type DecisionDependent
    """
    def matchups(self,participants):
        pairs = self.decision_ordering_pairs
        last_decision,last_ordering = pairs[0]
        yield pairs[0]
        
        last_action = last_decision.last_action
        last_payoff = last_decision(last_action)
        for decision_maker, ordering in pairs:
            decision = decision_maker([last_payoff[i] for i in last_ordering])
            yield (decision,ordering)
            last_decision,last_ordering = decision, ordering
            last_action = last_decision.last_action
            last_payoff = last_decision(last_action)

def BinaryDictatorDict(endowment = 0, cost = 1, benefit = 2):
    return {
        "keep": (endowment, 0),
        "give": (endowment-cost, benefit)
    }
def BinaryDictator():
    return Decision(BinaryDictatorDict())

class PrisonersDilemma(SymmetricGame):
    def __init__(self, endowment = 0, cost = 1, benefit = 2):
        payoffs = BinaryDictatorDict(endowment,cost,benefit)
        super(PrisonersDilemma,self).__init__(payoffs)

class UltimatumPropose(DecisionObserved):
    def __init__(self, endowment = 10):
        payoffs = {"keep {}/give {}".format(keep, give) : (keep, give) for keep, give in
                   ((endowment - give, give) for give in xrange(endowment))}
        super(UltimatumPropose,self).__init__(payoffs)

class UltimatumDecide(DecisionDependent):
    def __init__(self,proposed_payoff):
        payoffs = {"accept":(0,)*len(proposed_payoff),
                   "reject":tuple(-array(proposed_payoff))}
        super(UltimatumDecide,self).__init__(payoffs)

class UltimatumGame(DecisionDependentSeq):
    def __init__(self,endowment = 10):
        pairs = [(UltimatumPropose(endowment),[0,1]),
                 (UltimatumDecide,[1,0])]
        super(UltimatumGame,self).__init__(pairs)

class RepeatedGame(DecisionSeq):
    """
    Specified by a game and a number of repetitions
    this DecisionSeq plays the given game with a
    
    """
    def __init__(self,game,repetitions):
        self.game = game
        self.repetitions = repetitions

    def matchups(self,participants):
        game = self.game
        ordering = range(len(participants))
        return ((game,ordering) for _ in xrange(self.repetitions))
    
class RandomizedTournament(DecisionSeq):
    """
    This DecisionSeq plays the game with every adequately 
    sized subset of a given population of participants
    """
    def __init__(self,game):
        self.game = game

    def matchups(self, participants):
        matchups = itertools.combinations(xrange(len(participants)), self.game.N_players)
        game = self.game
        return ((game, matchup) for matchup in matchups)

def run_equivalent(repetitions):
    return RandomizedTournament(RepeatedGame(PrisonersDilemma(),repetitions)).play(agents,observers)
