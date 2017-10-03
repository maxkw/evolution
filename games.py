from __future__ import division
from utils import flip
from itertools import product, combinations, permutations,cycle
from numpy import array
from copy import copy, deepcopy
import numpy as np
from experiment_utils import fun_call_labeler
from inspect import getargspec
import random

COST = 1
BENEFIT = 3
ENDOWMENT = 0
ROUNDS = 10

def literal(constructor):
    """
    use this decorator for functions that generate playables
    this function names the playable after the function call that makes it
    """
    def call(*args,**kwargs):
        fun_call_string = fun_call_labeler(constructor,args,kwargs)['defined_call']
        call.__getargspec__ = constructor.__getargspec__
        ret = constructor(*args,**kwargs)
        ret.name = ret._name = fun_call_string
        return ret
    call.__name__ = constructor.__name__
    call.__getargspec__ = getargspec(constructor)
    return call

def implicit(constructor):
    """
    use this to correctly label functions that take as their first element another function
    and use args/kwargs to capture the arguments to said function while explicitly naming their own
    """
    def call(*args,**kwargs):
        const_call_data = fun_call_labeler(constructor,args,kwargs)
        fun_key, fun = const_call_data['defined_args'].items()[0]
        const_call_data['defined_args'][fun_key] = fun.__name__
        fun_call_data = fun_call_labeler(fun,[],const_call_data['undefined_args'])

        doubly_unused = [item for item in const_call_data['undefined_args'].items() if item in fun_call_data['undefined_args'].items()] 
        items = const_call_data['defined_args'].items()+fun_call_data['defined_args'].items()+doubly_unused
        ret = constructor(*args,**kwargs)
        ret.name = ret._name = constructor.__name__+"(%s)" % ", ".join(['%s' % items[0][1]]+["%s=%s" % item for item in items[1:]])
        
        return ret
    call.__name__ = constructor.__name__
    #call.__getargspec__ = getargspec(constructor)
    return call

class Playable(object):
    """
    this is the base class of all games
    these are defined by having the 'play' method
    
    this class is robust and accepts any decision that an agent accepts

    Note on naming:
    
    all Playable class instances should have a 'name' and '_name' variable
    whose value is the string representing the expression that generates it
    for example:
        Combinatorial(PrisonersDilemma(cost = COST, benefit = BENEFIT)).name has the value
        'Combinatorial(PrisonersDilemma(cost = COST, benefit = BENEFIT))'

    the idea is that the name perfectly specifies how the object came to be and how
    to recreate it.
    """
    def play(decision,participants,observers=[],tremble=0):
        """
        returns a dict containing
        the decision
        the final action
        the ids of the participants in the order they were given
        the ids of the observers (participants are always on this list)
        an array of the payoffs of each agent as ordered in participant_ids

        NOTE:
        participants are always added to observer list (with no doubling)
        """
        participant_ids = [participant.world_id for participant in participants]
        decider = participants[0]
        intention = decider.decide(decision,participant_ids)

        if flip(decision.tremble):
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
    def next_game(self):
        return self
    def __repr__(self):
        return self.name
    def __hash__(self):
        return hash(self.name)

"""
Observation Modifiers
The following classes cause agents to observe after the 'play' function is called

Keep this in mind when using observations on more complex playables:

PubliclyObserved(Combinatorial(PrisonersDilemma)) is a playable where observations happen only after all combinations of participants have played.

Combinatorial(PubliclyObserved(PrisonersDilemma)) is a playable where every pair of participants plays and then immediately observes the game.
"""

class RandomlyObserved(Playable):
    """
    randomly selects a specified percent of the provided observers
    percent is determined by self.observability, which must be set beforehand
    these observers are passed down into _play
    selected observers and all participants observe
    """
    def __init__(self,observability, playable):
        self.name = "RandomlyObserved(%s,%s)" % (observability, playable.name)
        self.observability = observability
        self.N_players = playable.N_players
        self.playable=playable

    def play(self, participants, observers = [], tremble=0):
        if self.observability < 1:
            # Sample from the list of possible observers
            observers = np.random.choice(observers,
                                         size=int(len(observers)*self.observability),
                                         replace=False)
        elif self.observability == 0:
            observers = []
            
        payoffs, observations, notes = self.playable.play(participants, observers, tremble)
        
        for observer in set(list(observers)+list(participants)):
            observer.observe(observations)
            
        return payoffs, observations, notes

class PrivatelyObserved(RandomlyObserved):
    def __init__(self,playable):
        super(PrivatelyObserved, self).__init__(0, playable)
        self.name = "PrivatelyObserved(%s)" % playable.name

class PubliclyObserved(RandomlyObserved):
    def __init__(self,playable):
        super(PubliclyObserved, self).__init__(1, playable)
        self.name = "PubliclyObserved(%s)" % playable.name

class ObservedByFollowers(Playable):
    def __init__(self,observability, playable):
        self.name = "ObservedByFollowers(%s,%s)" % (observability, playable.name)
        self.playable = playable
        self.N_players = playable.N_players
        self.observability = observability
        self.followers = dict()
    
    def play(self, participants, observers = [], tremble = 0):
        a_id = participants[0].world_id
        
        if a_id not in self.followers:
            self.followers[a_id] = np.random.choice(observers,size=int(len(observers)*self.observability),replace=False)
        observers = self.followers[a_id]

        payoffs, observations, notes = self.playable.play(participants, observers, tremble)

        for observer in set(list(observers)+list(participants)):
            observer.observe(observations)

        return payoffs, observations, notes

"""
Decisions
These are the only things Agents actually know how to deal with
every other class here orchestrates different ways to get a pool of
agents to actually play a decision. 
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
        return array(self.payoffs[action])



def BinaryDictatorDict(endowment = 0, cost = COST, benefit = BENEFIT):
    return {
        "keep": (endowment, 0),
        "give": (endowment-cost, benefit)
    }

def BinaryDictator(endowment = 0, cost = COST, benefit = BENEFIT, tremble = 0):
    """
    a 2-participant decision
    """
    decision = Decision(BinaryDictatorDict(endowment,cost,benefit))
    decision.tremble = tremble
    decision.name = decision._name = "BinaryDictator(%s)" % ",".join(map(str,[endowment,cost,benefit]))
    return decision

@literal
def GradatedBinaryDictator(endowment = ENDOWMENT, cost = COST, benefit = BENEFIT, intervals = 2, tremble = 0):
    ratio = cost/benefit
    benefits = np.linspace(0,benefit,intervals)
    costs = benefits*ratio
    decision_names = [str(p) for p in zip(endowment-costs,benefits)]
    decision_names[0] = 'keep'
    decision_names[-1] = 'give'
    decision_dict = dict(zip(decision_names, zip(endowment-costs,benefits)))
    decision = Decision(decision_dict)
    decision.tremble = tremble
    
    
    #decision._name = "GradatedBinaryDictator(%s)" % ",".join(map(str,[endowment,cost,benefit]))"
    return decision

@literal
def SocialDictator(endowment = ENDOWMENT, cost = COST, benefit = BENEFIT, intervals = 2, tremble = 0):
    cost = float(cost)
    benefit = float(benefit)
    max_d = benefit-cost
    max_r = cost/benefit
    ratios = np.linspace(0,max_r,intervals)
    differences = np.linspace(0,max_d,intervals)
    def new_cost(r,d):
        return d/(1-r)-d
    def new_benefit(r,d):
        return d/(1-r)

    payoffs = [(endowment-new_cost(r,d), new_benefit(r,d)) for d,r in zip(differences,ratios)]


    decision = Decision(dict((str(p),p) for p in payoffs))
    decision.tremble = tremble
    return decision

@literal
def TernaryDictator(endowment = 0, cost = COST, benefit = BENEFIT, tremble = 0):
    payoffs = [
        (endowment,0,0),
        (endowment-cost,benefit,0),
        (endowment-cost,0,benefit)
    ]
    decision = Decision(dict((str(p),p) for p in payoffs))
    decision.tremble = tremble
    return decision

@literal
def TernaryIgnore(endowment = 0, cost = COST, benefit = BENEFIT, tremble = 0):
    payoffs = [
        (endowment,0,0),
        (endowment-cost,benefit,0),
    ]
    decision = Decision(dict((str(p),p) for p in payoffs))
    decision.tremble = tremble
    return decision

"""
Decision Seqs
These are used to play multiple games in a row
the games played are independent from each other

though you can manually feed it pairs of games and orderings of players this is not
reccomended

the 'matchups' method is in charge of producing the pairs of games and players
that will be played

other classes overwrite 'matchups' to get more complicated behavior from this base
class
"""

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
    name = "DecisionSeq"
    def __init__(self, decision_ordering_pairs):
        self.decision_ordering_pairs = decision_ordering_pairs
        self.play_ordering_pairs = [(decision, ordering) for decision, ordering in self.decision_ordering_pairs]
        
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
        
        for game,ordering in self.matchups(participants):
            pay,obs,rec = game.play(participants[ordering],observers,tremble)
            payoffs[ordering] += pay
            extend_obs(obs)

        return payoffs,observations,[]
    _play = play

"""
Match Fixers

The following classes take in a game and automatically
slice a given pool of participants to the appropriate size
and feed them into the given game

NOTE:
these classes adopt the N_players attribute from their subgames
and they themselves use this attribute to fix their subgames

if fed themselves into a fixer they will be treated as if
the number of players for them is the same as their subgame

meaning that 
Combinatorial(Symmetric(BinaryDictator())).play(p)
and
Symmetric(BinaryDictator()).play(p)
will play the same number of games for the same p
"""
class CombinatorialMatchup(object):
    """
    the playable is played exactly once by every adequately sized 
    subset of the participants.
    
    NOTE:
    every subset plays exactly once
    best for games that handle their own ordering
    """
    def __init__(self,game):
        self.name = self._name = "Combinatorial("+game.name+")"
        self.game = game
        self.N_players = game.N_players
    def matchups(self, participants):
        matchups = list(combinations(xrange(len(participants)), self.game.N_players))
        np.random.shuffle(matchups)
        game = self.game
        for matchup in matchups:
            yield (game, list(matchup))

class Combinatorial(CombinatorialMatchup,DecisionSeq):
    pass
            
class SymmetricMatchup(object):
    """
    plays the game with every possible permutation of the participants
    
    
    NOTE:
    there will be redundancy if ordering does not matter.
    this is best for games where every position is different.
    """
    def __init__(self,game):
        self.game = game
        self.N_players = game.N_players

    def matchups(self, participants):
        matchups = list(permutations(xrange(len(participants)), self.game.N_players))
        np.random.shuffle(matchups)
        game = self.game
        for matchup in matchups:
            yield (game, list(matchup))

class Symmetric(SymmetricMatchup,DecisionSeq):
    pass

class SymmetricRecipients(DecisionSeq):
    def __init__(self,game):
        self.game = game
        self.N_players = game.N_players

    def matchups(self, participants):
        ids = set(xrange(len(participants)))
        matchups = []
        for i in ids:
            matchups.extend([(i)+p for p in permutations(ids-i)])
        np.random.shuffle(matchups)
        game = self.game
        for matchup in matchups:
            yield (game, list(matchup))

class CircularMatchup(object):
    def __init__(self,game):
        self.game = game
        self.name = self._name = "Circular("+game.name+")"
        self.N_players = game.N_players

    def matchups(self,participants):
        while True:
            indices = range(len(participants))
            np.random.shuffle(indices)
            
            matchups = zip(*[indices[i:]+indices[:i] for i in xrange(self.game.N_players)])
            #matchups = zip(indices,indices[1:]+indices[:1])

            playable = self.game
            for matchup in matchups:
                yield (playable, list(matchup))

class RandomMatching(DecisionSeq):
    def __init__(self,game):
        self.game = game
        self.name = self._name = "RandomlyMatching("+game.name+")"

    def matchups(self,participants):
        indices = range(len(participants))
        while True:
            game = self.game.next_game()
            N_players = game.N_players
            np.random.choice(indices, size=N_players,replace=False)
            yield game, np.random.choice(indices, size=N_players, replace=False)

class Circular(CircularMatchup, DecisionSeq):
    pass

class EveryoneDecidesMatchup(object):
    """
    plays the playable with every adequately sized subset of participants
    and every participant gets to be the first player once per subset

    NOTE:
    This fixer is meant to handle playables where only the decider matters
    in these games all payoffs to non-deciders are symmetrical and of equal
    cost to deciders
    """
    def __init__(self, playable):
        self.name = "EveryoneDecides(%s)" % playable.name
        self.playable = playable
        self.N_players = playable.N_players

    def matchups(self, participants):
        matchups = []; append = matchups.append
        size = self.N_players
        for combination in combinations(xrange(len(participants)), size):
            for i in range(n):
                append(combination[i:i+1]+combination[:i]+combination[i+1:size])
        np.random.shuffle(matchups)

        playable = self.playable
        for matchup in matchups:
            yield (playable, list(matchup))
            
class EveryoneDecides(EveryoneDecidesMatchup,DecisionSeq):
    pass

@literal
def PrisonersDilemma(endowment = 0, cost = COST, benefit = BENEFIT):
    return Symmetric(BinaryDictator(endowment,cost,benefit))



"""
Dependent Decisions

These are special Decisions whose characteristics are determined
by the events of a previous playable.

The first element in a sequence of these must be independent, of course.

NOTE:
although not enforced, it only makes sense that the playable that precedes a
dependent decision be observed by an agent if they are present for both.
"""

class DecisionDependent(Decision):
    """
    This type of decision is defined solely by a payoff

    this, along with DecisionDependentSeq are meant to be a framework
    for implementing things like the ultimatum game
    """
    pass

class DecisionDependentSeq(DecisionSeq):
    """
    This Decision Seq is made up of Observed decisions
    The first should be a Decision
    The rest must be of type DecisionDependent

    if this kind of DecisionSeq is itself observed it will be twice observed, as its 
    constituent decisions are observed
    """
    def __init__(self, decision_ordering_pairs):
        self.decision_ordering_pairs = [(d,array(o)) for d,o in decision_ordering_pairs] 
        
    def matchups(self,participants):
        pairs = self.decision_ordering_pairs
        last_decision,last_ordering = pairs[0]
        #last_ordering = array(last_ordering)
        yield pairs[0]#[0],array(pairs[0][1])
        
        last_action = last_decision.last_action
        last_payoff = last_decision(last_action)
        for decision_maker, ordering in pairs[1:]:
            # Check if this is a bug in terms of payoff ordering
            decision = decision_maker(last_payoff[last_ordering][ordering])
            yield (decision,ordering)
            last_decision,last_ordering = decision, ordering
            last_action = last_decision.last_action
            last_payoff = last_decision(last_action)

"""
Ultimatum Game

this game is built up of a 'Propose' and 'Decide' game
"""
class UltimatumPropose(Decision):
    def __init__(self, endowment = 10):
        payoffs = dict()
        self.name = "UltimatumProposal(%s)" % endowment
        for give in range(endowment+1):
            keep = endowment - give
            payoffs["keep %d/give %d" % (keep, give)] = (keep, give)

        super(UltimatumPropose,self).__init__(payoffs)

class UltimatumDecide(DecisionDependent):
    def __init__(self,proposed_payoff):
        self.name = "UltimatumDecide([%s])" % ",".join(map(str,proposed_payoff))
        payoffs = {"accept":(0,)*len(proposed_payoff),
                   "reject":tuple(-array(proposed_payoff))}
        super(UltimatumDecide,self).__init__(payoffs)

class UltimatumGame(DecisionDependentSeq):
    def __init__(self,endowment = 10):
        self.name = "UltimatumGame(%s)" % endowment
        pairs = [(UltimatumPropose(endowment),[0,1]),
                 (UltimatumDecide,[1,0])]
        super(UltimatumGame,self).__init__(pairs)

"""
Annotated Decision Sequences

these are used to collect data about the games mid-way
"""


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

    def play(self,participants,observers=None,tremble=0,notes={}):
        if observers is None:
            observers = participants
        

        #initialize accumulators
        observations = []
        payoffs = np.zeros(len(participants))
        record = [{
            'round':0,
            'payoff':payoffs,
            'beliefs': tuple(copy(getattr(agent, 'belief', None)) for agent in participants),
            'likelihoods': tuple(copy(getattr(agent, 'likelihood', None)) for agent in participants),
            'new_likelihoods': tuple(copy(getattr(agent, 'new_likelihoods', None)) for agent in participants)
        }]

        record = []

        #cache the dot references
        extend_obs = observations.extend
        extend_rec = record.append
        annotate = self.annotate

        
        extend_rec(annotate(participants,payoffs,[],[], notes))

        for game,ordering in self.matchups(participants):
            pay,obs,rec = game.play(participants[ordering],observers,tremble)
            new_payoffs = np.zeros(len(participants))
            new_payoffs[ordering] += pay
            payoffs += new_payoffs
            extend_rec(annotate(participants,new_payoffs,obs,rec,notes))
            extend_obs(obs)

        assert len(payoffs)==len(participants)
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
    def __init__(self, rounds, game):
        self.name = self._name= "Repeated("+str(rounds)+","+game.name+")"
        self.game = game
        self.rounds = rounds
        self.N_players = game.N_players
        self.current_round = 0

    def annotate(self,participants,payoff,observations,record,notes):
        note = {
            'round':self.current_round,
            'actions':tuple(observation[3] for observation in observations),
            'actors':tuple(observation[1] for observation in observations),
            'payoff': copy(payoff),
            'games':tuple(observation[0] for observation in observations),
            'beliefs': tuple(copy(getattr(agent, 'belief', None)) for agent in participants),
            'likelihoods' :tuple(deepcopy(getattr(agent,'likelihood', None)) for agent in participants),
            'new_likelihoods':tuple(copy(getattr(agent, 'new_likelihoods', None)) for agent in participants),
            }
        note.update(notes)
        return note
    
    def matchups(self,participants):
        ordering = range(len(participants))
        for game_round in xrange(1,self.rounds+1):
            self.current_round = game_round
            yield self.game, ordering

class Annotated(AnnotatedDS):
    def __init__(self,rounds, game):
        self.rounds = rounds
        self.name = self._name= "Annotated("+game.name+")"
        self.game = game
        #self.N_players = game.N_players
        self.current_round = 0

    def annotate(self,participants,payoff,observations,record,notes):
        note = {
            'round':self.current_round,
            'actions':tuple(observation[3] for observation in observations),
            'actors':tuple(observation[1] for observation in observations),
            'payoff': copy(payoff),
            'games':tuple(observation[0] for observation in observations),
            'beliefs': tuple(copy(getattr(agent, 'belief', None)) for agent in participants),
            'likelihoods' :tuple(deepcopy(getattr(agent,'likelihood', None)) for agent in participants),
            'new_likelihoods':tuple(copy(getattr(agent, 'new_likelihoods', None)) for agent in participants),
            }

        note.update(notes)
        return note
    
    def matchups(self,participants):
        for game_round, thing in zip(xrange(1,self.rounds+1), cycle(self.game.matchups(participants))):
            self.current_round = game_round
            yield thing
            
@literal
def AnnotatedCircular(game):
    return Annotated(Circular(game))

class IndefiniteHorizonGame(DecisionSeq):
    def __init__(self,gamma,playable):
        self.name = self._name = "IndefiniteHorizon(%s,%s)" % (gamma,playable.name)
        self.playable = playable
        self.gamma = gamma
        
    def matchups(self,participants):
        """
        yields the game/ordering pair once
        subsequently yields with probability gamma
        """
        game = self.playable
        gamma = self.gamma
        ordering = range(len(participants))
        yield game,ordering
        while flip(gamma):
            yield game, ordering

class thunk(object):
    def __init__(self,fun,*args,**kwargs):
        self.__dict__.update(locals())
    def __call__(self):
        return self.fun(*self.args,**self.kwargs)

class const(object):
    def __init__(self,val):
        self.val = val
    def __call__(self):
        return self.val


class RandomlyChosen(Playable):
    def __init__(self,*playables):
        self._playables = playables
        self.N_players = N = playables[0].N_players
        assert all(p.N_players == N for p in playables)
        self.name = "RandomlyChosen(%s)" % ", ".join([p.name for p in playables])
    def play(self, *args, **kwargs):
        return random.choice(self._playables).play(*args,**kwargs)


class Dynamic(Playable):
    """
    takes a playable-making function
    and a thunk that generates parameters for it
    whenever it's 'play' method is called, it generates
    new arguments and serves up a playable made with them
    """
    def __init__(self,playable_constructor,gen):
        self.constructor = playable_constructor
        self.arg_gen = gen
        instance = playable_constructor(**gen())
        self.N_players = instance.N_players
        self.shuffle_game()

    def new(self):
        return self.constructor(**self.arg_gen())

    def shuffle_game(self):
        self.current_version = self.constructor(**self.arg_gen())
        return self.current_version

    def play(self,participants,observers=[], tremble=0):
        playable = self.constructor(**self.arg_gen())
        self.shuffle_game()
        return playable.play(participants, observers, tremble)

class OrGame(Playable):
    def __init__(self,*games):
        self.name = "OrGame(%s)" % ", ".join(g.name for g in sorted(games))
        self.games = games
        self.next()

    def next_game(self):
        g = self.current_game = np.random.choice(self.games)
        return g

    def play(self, participants, observers = [], tremble=0):
        playable = self.current_game
        self.next_game()
        return playable.play(participants, observers, tremble)

def exponential(scale,gamma=1):
    return np.random.exponential(gamma)*scale

def constant(val,**kwargs):
    return val

def poisson():
    pass
def normal():
    pass

def dict_map(f,d, **kwargs):
    """
    given a function and a dictionary of argnames to argvals
    (and optionally any keyword args that also correspond to arg name-val pairs)
    the function is called using the keyword args as default values and overwriting them with the provided dict
    """
    ret_dict = {}
    for key,val in d.iteritems():
        try:
            ret_dict[key] = f(**dict(kwargs,**val))
        except:
            ret_dict[key] = f(val,**kwargs)
    return ret_dict

def dist_apply(dist,**kwargs):
    """the same as 'apply' but does not accept positional parameters"""
    return dist(**kwargs)

def dist_dict(d,dist,**kwargs):

    ret_dict = {}
    for key,val in d.iteritems():
        try:
            ret_dict[key] = dist_apply(**dict(kwargs,**val))
        except:
            ret_dict[key] = dist(val,**kwargs)
    return ret_dict

def thunk_apply(func,**kwargs):
    """
    given some function f and some arguments for it
    make an object t that when called feeds the arguments to f
    and returns the result
    a new call to f is made each time t is called
    if f and its arguments define a distribution
    then a call to t is a sampling of that distribution
    """
    return thunk(func,**kwargs)

def thunk_dict(d,func,**kwargs):
    """
    this function is meant to return a dictionary with arguments for some function F
    it is given a dictionary A whose keys are argnames and whose vals are dicts that are
    used to create the values that will actually be passed down to F.


    This functon takes a function D as one of it's parameters.

    This function is simple but the logic is obtuse.
    Basically, the dictionaries at the second level can provide their own distribution and arguments,
    if not then that dictionary is assumed to contain arguments for a distribution found at the first level.

    In greater detail.

    If some key K in A maps to a dict B that doesn't have 'dist' as a key, then we assume that B
    is a dict of valid arguments for D, and we let K map to D(**B) in the return dict.

    Let B = A[K] once more. If 'dist' is a key in B, then we assume B['dist'] is some function H.
    If C is the dict B without the key 'dist' then we assume that C is valid mapping of arguments for H.
    We let K map to H(**C) in the return dict.

    """
    ret_dict = {}
    for key,val in d.iteritems():
        try:
            ret_dict[key] = thunk_apply(**dict(kwargs,**val))
        except:
            ret_dict[key] = thunk(func,val,**kwargs)
    return ret_dict

def dict_thunker(d):
    """
    given a dict where all values are thunks
    generates a dictionary with the same key but
    where the value is generated by calling the thunk
    """
    return {key:val() for key,val in d.iteritems()}

@implicit
def Randomly(game,*args,**kwargs):
    """
    if any explicitly named argument of 'game' is passed in as a dict containing the 'dist' key
    then any additional parameters required by the distribution should also be provided either
    in that same dictionary or in the greater function call
    """
    call_data = fun_call_labeler(game,args,kwargs)
    game_args = call_data['defined_args']

    dist_args = dict({'func':constant},**call_data['undefined_args'])
    return Dynamic(game,thunk(dict_thunker,thunk_dict(game_args,**dist_args)))
                   #thunk(dist_dict,game_args,**dist_args))game = Randomly(PrisonersDilemma)



"""
example:

Randomly(GradatedBinaryDictator,
         cost = dict(func = exponential, gamma = 5, scale = 1),
         benefit = dict(func = normal, mean = 5, variance = 2),
         intervals = 5)
"""


@implicit
def Exponential(game,gamma=1,*args,**kwargs):
    return Randomly(game, func = exponential, gamma = gamma, *args, **kwargs)


"""
example:

Exponential(PrisonersDilemma, cost = {'gamma':9,'scale':3}, benefit = {'gamma':3,'scale':4})

"""

#@literal
def RepeatedDynamicPrisoners(rounds = ROUNDS, endowment = 0, cost = COST, benefit = BENEFIT, gamma = 1):
    return Repeated(rounds,PrivatelyObserved(Exponential(PrisonersDilemma)))
#return Repeated(rounds,PrivatelyObserved(DynamicPD()))

def RepeatedSequentialBinary(rounds = ROUNDS, visibility = "private"):
    BD = BinaryDictator(cost = COST, benefit = BENEFIT)
    return Repeated(rounds,PrivatelyObserved(Symmetric(BD)))

@literal
def RepeatedPrisonersTournament(rounds = ROUNDS, cost=COST, benefit=BENEFIT, tremble = 0, intervals = 2, **kwargs):
    visibility = "private"
    observability = .5
    PD = Symmetric(GradatedBinaryDictator(cost = cost, benefit = benefit, intervals=intervals, tremble = tremble))

    if visibility == "private":
        PD.tremble = kwargs.get('tremble', 0)
        g =  Repeated(rounds, PrivatelyObserved(PD))
        g.tremble = kwargs.get('tremble', 0)
        return g
    if visibility == "random":
        return Repeated(rounds, PubliclyObserved(Combinatorial(RandomlyObservable(observability,PD))))
    if visibility == "public":
        return Repeated(rounds, PubliclyObserved(PD))


@literal
def IndirectReciprocity(rounds = ROUNDS, cost = COST, benefit = BENEFIT, tremble = 0, observability = 1, intervals = 2, **kwargs):
    bd = GradatedBinaryDictator(cost = cost, benefit = benefit, intervals = intervals, tremble = tremble)
    g = Repeated(rounds, Circular(RandomlyObserved(observability, bd)))
    return g

@literal
def ExponentialIndirectReciprocity(rounds = ROUNDS, cost = COST, benefit = BENEFIT, tremble = 0, observability = 1, intervals = 2, **kwargs):
    bd = Randomly(GradatedBinaryDictator,
                     cost = dict(func = exponential, gamma = 1, scale = cost),
                     benefit = dict(func = exponential, gamma = 1, scale = benefit),
                     intervals = intervals,
                     tremble = tremble)
    g = Repeated(rounds, Circular(RandomlyObserved(observability, bd)))
    return g


@literal
def GradatedTournament(rounds = ROUNDS, cost = COST, benefit = BENEFIT, tremble = 0, intervals = 2, **kwargs):
    args = dict(cost=cost, benefit = benefit, tremble = tremble, intervals = intervals)
    game = Repeated(rounds, PubliclyObserved(Symmetric(GradatedBinaryDictator(**args))))
    return game


@literal
def SocialTournament(rounds = ROUNDS, cost = COST, benefit = BENEFIT, tremble = 0, observability = 1, intervals = 2, followers = True, **kwargs):
    bd = GradatedBinaryDictator(cost = cost, benefit = benefit, intervals = intervals, tremble = tremble)
    
    if followers:
        game = Annotated(rounds, Circular(ObservedByFollowers(observability,bd)))
    else:
        game = Annotated(rounds, Circular(RandomlyObserved(observability,bd)))

    return game

if __name__ == "__main__":
    from agents import WeAgent, Puppet
    import agents
    from params import default_genome
    
    puppets = array([Puppet("Alpha"),Puppet("Beta"),Puppet("C")])
    

    game = TernaryTournament(10)#RepeatedDynamicPrisoners(10)
    print game
    print hash(game)
    payoff,history,records = game.play(puppets)
    print "Final Payoff:",payoff
    print len(list(set(g for g,a,b,c in history)))
  
