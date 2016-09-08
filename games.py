from utils import flip, randomly_chosen
from itertools import product,combinations,permutations
from numpy import array
from copy import copy
import numpy as np

class Playable(object):
    """
    this is the base class of all games
    these are defined by having the 'play' method
    
    this class is robust and accepts any decision that an agent accepts

    Note on naming:
    
    all Playable class instances should have a 'name' and '_name' variable
    whose value is the string representing the expression that generates it
    for example:
        Combinatorial(PrisonersDilemma(cost = 1, benefit = 3)).name has the value
        'Combinatorial(PrisonersDilemma(cost = 1, benefit = 3))'

    Playable Classes and functions that take in and return Playables as parameters 
    should likewise set their own names, or that of the playable, to the appropriate 
    name.

    the idea is that the name perfectly specifies how the object came to be and how
    to recreate it.
    """
    def _play(decision,participants,observers=[],tremble=0):
        """
        returns a dict containing
        the decision
        the final action
        the ids of the participants in the order they were given
        the ids of the observers (participants are always on this list)
        an array of the payoffs of each agent as ordered in participant_ids
        """
        participant_ids = [participant.world_id for participant in participants]
        decider = participants[0]
        intention = decider.decide(decision,participant_ids)
        
        if flip(tremble):
            action = np.random.choice(decision.actions)
        else:
            action = intention

        payoffs = copy(decision(action))
            
        observer_ids = tuple(observer.world_id for observer in set(list(observers)+list(participants)))
        
        observations = [{
            'game':decision,
            'action':action,
            'participant_ids':array(participant_ids),
            'observer_ids':array(observer_ids),
            'payoffs':array(payoffs)}]
        
        observations = [(decision,participant_ids,observer_ids,action)]
        return payoffs, observations, None
 
    play = _play
    """
    the following functions are wrappers around the '_play' method
    these internally call '_play' but extract and process the observations
    before passing the results back up.
    
    DO NOT USE ANY OF THESE DIRECTLY, NOT EVEN '_observe'
    """
    def _publicly_observed_play(self,participants,observers=[], tremble=0):
        """
        passes observer list down to _play
        observers and participants observe
        """
        payoffs, observations, notes = self._play(participants,observers,tremble)
        for observer in set(list(observers)+list(participants))):       
            observer.observe_k(observations,observer.genome['RA_K'], tremble)
        return payoffs, observations, notes
    
    def _privately_observed_play(self,participants,observers=[], tremble=0):
        """
        calls _play with no observers
        participants observe
        """
        payoffs, observations,notes = self._play(participants,[],tremble)
        for observer in list(participants):
            observer.observe_k(observations,observer.genome['RA_K'], tremble)
        return payoffs, observations, notes

    def _randomly_observed_play(self,participants,observers=[], tremble=0):
        """
        randomly selects a specified percent of the provided observers
        percent is determined by self.observability, which must be set beforehand
        these observers are passed down into _play
        selected observers and all participants observe
        """
        observers = randomly_chosen(self.observability,observers))
        payoffs, observations, notes = self._play(participants,observers,tremble)
        for observer in set(list(observers)+list(participants))):
            observer.observe_k(observations,observer.genome['RA_K'], tremble)
        return payoffs, observations, notes
    

    def _observing(self,observation_type = False, observability=.5):
        """
        overwrites the 'play' function according to provided parameters
        changes 'name' attribute with every call.
        """
        if not observation_type:
            self.play = self._play
        if observation_type == 'public':
            self.name = "PubliclyObserved(%s)" % self.name
            self.play = self._publicly_observed_play
        if observation_type == 'private':
            self.name = "PrivatelyObserved(%s)" % self.name
            self.play = self._privately_observed_play
        if observation_type == 'random':
            assert 1>=observability and observability>=0
            self.name = "RandomlyObserved(%s,%s)" % (observability,self.name)
            self.observability = observability
            self.play = self._randomly_observed_play
        return self

"""
Use these functions to set the observability of a Playable

The following are pretty wrappers for setting the observability of Playables
these should never be nested with each other as they'll overwrite each other.
"""

def PrivatelyObserved(playable):
    return playable._observing('private')
def PubliclyObserved(playable):
    return playable._observing('public')
def RandomlyObserved(observability,playable):
    return playable._observing('random',observability)

"""
Keep this in mind when using observations on more complex playables:

PubliclyObserved(Combinatorial(PrisonersDilemma)) is a playable where observations happen only after all combinations of participants have played.

Combinatorial(PubliclyObserved(PrisonersDilemma)) is a playable where every pair of participants plays and then immediately observes the game.

"""

class Decision(Playable):
    """
    A decision is defined by a payoffDict
    
    Makes a playable object from a dictionary whose values are arrays
    all values are assumed to have the same length
    
    when called using 'play' the first agent will choose from among the keys
    the nth payoff then corresponds to the nth agent in the provided list
    """
    name = _name = "Decision"
    def __init__(self,payoffDict):
        actions = payoffDict.keys()
        self.N_players = len(payoffDict.values()[0])
        self.actions = actions
        self.action_lookup = dict(map(reversed,enumerate(actions)))
        self.payoffs = payoffDict
        
    def __call__(self,action):
        self.last_action = action
        return self.payoffs[action]


class DecisionSeq(Playable):
    """
    A DecisionSequence is specified by:
    A sequence of Decision/order pairs.
    where an 'ordering' is actually the ordered indices of the players that will
    participate in the corresponding decision.

    The length of an ordering must be the same as the number of players for the 
    corresponding game

    games may have different numbers of players

    when the 'play' method is called, it must be called with a list of participants
    whose length is at least one number larger than the largest number in any
    ordering.

    Classes that inherit from DecisionSeq will mostly overwrite the 'matchups' method
    """
    name = _name = "DecisionSeq"
    def __init__(self, decision_ordering_pairs):
        self.decision_ordering_pairs = decision_ordering_pairs
        self.play_ordering_pairs = [(decision.play, ordering) for decision, ordering in self.decision_ordering_pairs]
        
    def matchups(self,participants):
        """
        this simply returns an iterator whose elements are the
        (decision.play,ordering) tuples that will be played

        note that the first element is a 'play' function and not the decision itself.

        this method is meant to be overwritten by child-classes
        """
        return iter(self.play_ordering_pairs)
    
    def play(self,participants,observers=[],tremble=0):
        """
        takes in at least a list of participants whose length is at least
        one number larger than the largest number in an ordering used at
        initialization

        what it does:
        Repeatedly calls the 'play_decisions' provided by 'matchups' using the 
        players whose indices are given by 'ordering'

        what it returns:
        the running tally of all payoffs accumulated by participants accross
        all decisions played
        
        a list of observations in the order in which they occurred.
        
        an empty list for compatibility with annotations
        """
        #initialize accumulators
        observations = []
        payoffs = np.zeros(len(participants))

        #cache the dot references
        extend_obs = observations.extend
        
        for play_decision,ordering in self.matchups(participants):
            pay,obs,rec = play_decision(participants[ordering],observers,tremble)
            payoffs[ordering] += pay
            extend_obs(obs)

        return payoffs,observations,[]
    _play = play
            
class SymmetricGame(DecisionSeq):
    """
    plays a decision with every meaningful permutation of deciders and non-deciders
    this game assumes that the order of non-deciders does not matter
    and that all decisions have a single decider
    """
    name = _name = "SymmetricGame"
    def __init__(self, payoffDict):
        N = len(payoffDict.values()[0])
        orderings = (range(n,n+1)+range(n)+range(n+1,N) for n in xrange(N))
        decisions = [(Decision(payoffDict),ordering) for ordering in orderings]
        self.N_players = N
        super(SymmetricGame,self).__init__(decisions)

class DecisionDependent(Decision):
    """
    This type of decision is defined solely by a payoff

    this, along with DecisionDependentSeq are meant to be a framework
    for implementing things like the ultimatum game
    """
    pass
    
class DecisionDependentSeq(DecisionSeq):
    """
    Can't remember if this works. NOT TESTED.
    
    This Decision Seq is made up of Observed decisions
    The first should be a Decision
    The rest must be of type DecisionDependent

    if this kind of DecisionSeq is itself observed it will be twice observed, as its 
    constituent decisions are observed
    """
    def __init__(self, decision_ordering_pairs):
        self.decision_ordering_pairs = decision_ordering_pairs
        
    def matchups(self,participants):
        pairs = self.decision_ordering_pairs
        last_decision,last_ordering = pairs[0]
        yield pairs[0]
        
        last_action = last_decision.last_action
        last_payoff = last_decision(last_action)
        for decision_maker, ordering in pairs[1:]:
            # Check if this is a bug in terms of payoff orderings
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
def BinaryDictator(endowment = 0, cost = 1, benefit = 2):
    """
    a 2-participant decision
    """
    decision = Decision(BinaryDictatorDict(endowment,cost,benefit))
    decision.name = decision._name = "BinaryDictator(%s)" % ",".join(map(str,[endowment,cost,benefit]))
    return decision

class PrisonersDilemma(SymmetricGame):
    """
    a 2-player game
    """
    def __init__(self, endowment = 0, cost = 1, benefit = 2):
        payoffs = BinaryDictatorDict(endowment,cost,benefit)
        super(PrisonersDilemma,self).__init__(payoffs)
        self.name = self._name = "PrisonersDilemma(%s)" % ",".join(map(str,[endowment,cost,benefit]))

class UltimatumPropose(Observed,Decision):
    def __init__(self, endowment = 10):
        payoffs = dict()
        for give in range(endowment):
            keep = endowment - give
            payoffs["keep %d/give %d" % (keep, give)] = (keep, give)
        
        # payoffs = {"keep {}/give {}".format(keep, give) : (keep, give) for keep, give in

        super(UltimatumPropose,self).__init__(payoffs)

class UltimatumDecide(DecisionDependent):
    def __init__(self,proposed_payoff):
        payoffs = {"accept":(0,)*len(proposed_payoff),
                   "reject":tuple(-array(proposed_payoff))}
        # FIXME: This is probably going to be a bug since the super is not defined. Solution is to get rid of __init__ in DecisionDependent?
        super(UltimatumDecide,self).__init__(payoffs)

class UltimatumGame(DecisionDependentSeq):
    def __init__(self,endowment = 10):
        pairs = [(UltimatumPropose(endowment),[0,1]),
                 (UltimatumDecide,[1,0])]
        super(UltimatumGame,self).__init__(pairs)

class AnnotatedDS(DecisionSeq):
    """
    must define a method self.annotate that takes participants, payoffs,observations, and records
    its results will be appended to the record and passed up

    except for the 'annotate' method, this works exactly the same as DecisionSeq
    
    This kind of sequence is used for collecting more data about the games.
    annotations happen AFTER the play function is called
    """
    def annotate(self,participants,payoff,observations,record):
        raise NotImplementedError
    
    def play(self,participants,observers=[],tremble=0):
        #initialize accumulators
        observations = []
        record = []
        payoffs = np.zeros(len(participants))

        #cache the dot references
        extend_obs = observations.extend
        extend_rec = record.append
        annotate = self.annotate
        
        for play_decision,ordering in self.matchups(participants):
            pay,obs,rec = play_decision(participants[ordering],observers,tremble)
            payoffs[ordering] += pay
            #print payoffs
            extend_rec(annotate(participants,payoffs,obs,rec))
            extend_obs(obs)

        return payoffs,observations,record
    
    _play = play


class Repeated(AnnotatedDS):
    """
    Specified by a game and a number of repetitions
    
    this class' annotations contain: the number of times played, the players, the actions, the running payoff at the moment, a copy of all agent's beliefs and likelihoods after each game.
    
    actions in an annotation correspond to all actions taken in sub-games this round

    payoffs, beliefs, likelihoods are ordered according to the ordering of players
    
    note that if the repeated game is, for example, a sequence, then the beliefs and likelihoods are those after the entire sequence has been played. if the sequence is not observed they will not change.
    """
    def __init__(self,repetitions,game):
        self.name = self._name= "Repeated("+str(repetitions)+","+game.name+")"
        self.game = game
        self.repetitions = repetitions
        self.N_players = game.N_players

    def annotate(self,participants,payoff,observations,record):
        note = {
            'round':self.current_round,
            'players':tuple(participants),
            'actions':tuple(observation[3] for observation in observations),
            'payoff': copy(payoff),
            'belief': tuple(copy(agent.belief) for agent in participants),
            'likelihood' :tuple(copy(agent.likelihood) for agent in participants),
            }
        return note
    
    def matchups(self,participants):
        game = self.game
        ordering = range(len(participants))
        for game_round in xrange(1,self.repetitions+1):
            self.current_round = game_round
            yield game.play,ordering
    
class IndefiniteHorizonGame(DecisionSeq):
    def __init__(self,game,gamma):
        self.name = self._name = "IndefiniteHorizon(%s,%s)" % (game.name,gamma)
        self.game = game
        self.gamma = gamma
        
    def matchups(self,participants):
        """
        yields the game/ordering pair once
        subsequently yields with probability gamma
        """
        game = self.game
        gamma = self.gamma
        ordering = range(len(participants))
        yield game,ordering
        while flip(gamma):
            yield game,ordering
            
class Combinatorial(DecisionSeq):
    """
    This DecisionSeq plays the game with every adequately 
    sized subset of a given population of participants
    
    games used here should handle meaningful orderings themselves.

    matchups are randomized, so games with observations will not be biased.
    
    """
    def __init__(self,game):
        self.name = self._name = "Combinatorial("+game.name+")"
        self.game = game
        self.N_players = game.N_players
    def matchups(self, participants):
        matchups = list(combinations(xrange(len(participants)), self.game.N_players))
        np.random.shuffle(matchups)
        play = self.game.play
        for matchup in matchups:
            yield (play, list(matchup))
            
class Symmetric(DecisionSeq):
    """
    plays the game with every possible permutation of the participants
    there will be redundancy if ordering does not matter.

    this is best for games where every position is different.
    
    matchups in this class are not randomized.
    keep in mind if the game has observed components.
    """
    def __init__(self,game):
        self.game = game
        self.N_players = game.N_players

    def matchups(self, participants):
        matchups = permutations(xrange(len(participants)), self.game.N_players)
        #np.random.shuffle(list(matchups))
        play = self.game.play
        for matchup in matchups:
            yield (play, list(matchup))

def RepeatedPrisonersTournament(rounds = 10,visibility = "private"):
    PD = 
    if visibility == "private":
        return Repeated(rounds,PrivatelyObserved(Combinatorial(PrisonersDilemma(cost = 1, benefit = 3))))
    if visibility == "random":
        return Repeated(rounds,PubliclyObserved(Combinatorial(RandomlyObserved(.5,PrisonersDilemma(cost = 1, benefit = 3)))))
    if visibility == "public":
        return Repeated(rounds,PubliclyObserved(Combinatorial(PrisonersDilemma(cost = 1, benefit = 3))))
