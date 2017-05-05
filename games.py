from utils import flip, randomly_chosen
from itertools import product,combinations,permutations
from numpy import array
from copy import copy,deepcopy
import numpy as np
from experiment_utils import fun_call_labeler
from inspect import getargspec

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
        Combinatorial(PrisonersDilemma(cost = 1, benefit = 3)).name has the value
        'Combinatorial(PrisonersDilemma(cost = 1, benefit = 3))'

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

TO DO:
Distinguish Observability vs Observation
These modules force observation. What about when I want to change the observer list
at the bottom level, but no actually observe?
"""
class PrivatelyObserved(Playable):
    """
    calls _play with no observers
    participants observe
    """
    def __init__(self,playable):
        self.name = "PrivatelyObserved(%s)" % playable.name
        self.N_players = playable.N_players
        self.playable=playable

    def play(self,participants,observers=[], tremble=0):
        payoffs, observations,notes = self.playable.play(participants,[],tremble)
        for observer in list(participants):
            observer.observe(observations)
        return payoffs, observations, notes

class PubliclyObserved(Playable):
    """
    passes complete observer list to _play
    observers and participants observe
    """
    def __init__(self,playable):
        self.name = "PubliclyObserved(%s)" % playable.name
        self.N_players = playable.N_players
        self.playable=playable

    def play(self,participants,observers=[], tremble=0):
        payoffs, observations, notes = self.playable.play(participants,observers,tremble)
        for observer in set(list(observers)+list(participants)):       
            observer.observe(observations)
        return payoffs, observations, notes
    
class RandomlyObserved(Playable):
    """
    randomly selects a specified percent of the provided observers
    percent is determined by self.observability, which must be set beforehand
    these observers are passed down into _play
    selected observers and all participants observe
    """
    def __init__(self,observability,playable):
        self.name = "RandomlyObserved(%s,%s)" % (observability, playable.name)
        self.observability = observability
        self.N_players = playable.N_players
        self.playable=playable

    def play(self,participants,observers=[], tremble=0):
        observers = randomly_chosen(self.observability,observers)
        payoffs, observations, notes = self.playable.play(participants,observers,tremble)
        for observer in set(list(observers)+list(participants)):
            observer.observe_k(observations,observer.genome['RA_K'], tremble)
        return payoffs, observations, notes
    
class RandomlyObservable(Playable):
    def __init__(self,observability,playable):
        self.name = "RandomlyObservable(%s,%s)" % (observability, playable.name)
        self.observability = observability
        self.N_players = playable.N_players
        self.playable=playable

    def play(self,participants,observers=[], tremble=0):
        observer_subset = randomly_chosen(self.observability,observers)
        return self.playable.play(participants,observer_subset,tremble)
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
def PrisonersDilemma(endowment = 0, cost = 1, benefit = 3):
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

    def play(self,participants,observers=[],tremble=0,notes={}):
        #initialize accumulators
        observations = []
        record = []
        payoffs = np.zeros(len(participants))

        #cache the dot references
        extend_obs = observations.extend
        extend_rec = record.append
        annotate = self.annotate

        extend_rec(annotate(participants,payoffs,[],[],notes))

        for game,ordering in self.matchups(participants):
            pay,obs,rec = game.play(participants[ordering],observers,tremble)
            payoffs[ordering] += pay
            extend_rec(annotate(participants,payoffs,obs,rec,notes))
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
        self.current_round = 0

    def annotate(self,participants,payoff,observations,record,notes):
        note = {
            'round':self.current_round,
            'players':tuple(deepcopy(agent) for agent in participants),
            'actions':tuple(observation[3] for observation in observations),
            'actors':tuple(observation[1] for observation in observations),
            'payoff': copy(payoff),
            #'belief': tuple(copy(agent.belief) for agent in participants),
            #'likelihood' :tuple(copy(agent.likelihood) for agent in participants),
            }
        note.update(notes)
        return note
    
    def matchups(self,participants):
        ordering = range(len(participants))
        for game_round in xrange(1,self.repetitions+1):
            self.current_round = game_round
            yield self.game,ordering
    
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
            yield game,ordering

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

    def new(self):
        return self.constructor(**self.arg_gen())

    def play(self,participants,observers=[], tremble=0):
        playable = self.constructor(**self.arg_gen())
        return playable.play(participants,observers, tremble)


def exponential(scale,gamma=1):
    return np.random.exponential(gamma)*scale
def constant(val,**kwargs):
    return val

def dict_map(f,d, **kwargs):
    ret_dict = {}
    for key,val in d.iteritems():
        try:
            ret_dict[key] = f(**dict(kwargs,**val))
        except:
            ret_dict[key] = f(val,**kwargs)
    return ret_dict

def dist_apply(dist,**kwargs):
    return dist(**kwargs)

def dist_dict(d,dist,**kwargs):
    ret_dict = {}
    for key,val in d.iteritems():
        try:
            ret_dict[key] = dist_apply(**dict(kwargs,**val))
        except:
            ret_dict[key] = dist(val,**kwargs)
    return ret_dict

def thunk_apply(dist,**kwargs):
    return thunk(dist,**kwargs)

def thunk_dict(d,dist,**kwargs):
    ret_dict = {}
    for key,val in d.iteritems():
        try:
            ret_dict[key] = thunk_apply(**dict(kwargs,**val))
        except:
            ret_dict[key] = thunk(dist,val,**kwargs)
    return ret_dict

def dict_thunker(d):
    """
    given a dict where all values are thunks
    generates a dictionary with the same key but
    where the value is generated by running the thunk
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

    dist_args = dict({'dist':constant},**call_data['undefined_args'])
    return Dynamic(game,thunk(dict_thunker,thunk_dict(game_args,**dist_args)))
                   #thunk(dist_dict,game_args,**dist_args))game = Randomly(PrisonersDilemma)

@implicit
def Exponential(game,gamma=1,*args,**kwargs):
    return Randomly(game, dist = exponential, gamma = gamma, *args, **kwargs)


"""
example:

Exponential(PrisonersDilemma, cost = {'gamma':9,'scale':3}, benefit = {'gamma':3,'scale':4})

"""

#@literal
def RepeatedDynamicPrisoners(rounds = 10, endowment = 0, cost = 1, benefit = 3, gamma = 1):
    return Repeated(rounds,PrivatelyObserved(Exponential(PrisonersDilemma)))
    #return Repeated(rounds,PrivatelyObserved(DynamicPD()))

def RepeatedSequentialBinary(rounds = 10, visibility = "private"):
    BD = BinaryDictator(cost = 1, benefit = 3)
    return Repeated(rounds,PrivatelyObserved(Symmetric(BD)))
@literal
def RepeatedPrisonersTournament(rounds = 10, cost=1, benefit=3,**junk):
    visibility = "private"
    observability = .5
    PD = PrisonersDilemma(cost = cost, benefit = benefit)
    if visibility == "private":
        return Repeated(rounds, PrivatelyObserved(PD))
    if visibility == "random":
        return Repeated(rounds, PubliclyObserved(Combinatorial(RandomlyObservable(observability,PD))))
    if visibility == "public":
        return Repeated(rounds, PubliclyObserved(PD))

if __name__ == "__main__":
    from indirect_reciprocity import ReciprocalAgent
    from params import default_genome
    
    from indirect_reciprocity import Puppet
    puppets = array([Puppet("Alpha"),Puppet("Beta"),Puppet("C")])
    agents = array([ReciprocalAgent(default_genome(),world_id = n) for n in range(3)])
    game = RepeatedDynamicPrisoners(10)
    print game
    print hash(game)
    payoff,history,records = game.play(agents)
    print "Final Payoff:",payoff
    print len(list(set(g for g,a,b,c in history)))
