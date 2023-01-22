from collections import OrderedDict
from utils import flip
from itertools import combinations, cycle, permutations
from numpy import array
from copy import copy, deepcopy
import numpy as np
from experiment_utils import fun_call_labeler
from inspect import getargspec

# import random
import itertools

COST = np.NaN
BENEFIT = np.NAN
ENDOWMENT = 0
ROUNDS = np.NaN


def literal(constructor):
    """
    use this decorator for functions that generate playables
    this function names the playable after the function call that makes it
    """

    def call(*args, **kwargs):
        fun_call_string = fun_call_labeler(constructor, args, kwargs)["defined_call"]
        call.__getargspec__ = constructor.__getargspec__
        ret = constructor(*args, **kwargs)
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

    def call(*args, **kwargs):
        const_call_data = fun_call_labeler(constructor, args, kwargs)
        fun_key, fun = list(const_call_data["defined_args"].items())[0]
        const_call_data["defined_args"][fun_key] = fun.__name__
        fun_call_data = fun_call_labeler(fun, [], const_call_data["undefined_args"])

        doubly_unused = [
            item
            for item in list(const_call_data["undefined_args"].items())
            if item in list(fun_call_data["undefined_args"].items())
        ]
        items = (
            list(const_call_data["defined_args"].items())
            + list(fun_call_data["defined_args"].items())
            + doubly_unused
        )
        ret = constructor(*args, **kwargs)
        ret.name = ret._name = constructor.__name__ + "(%s)" % ", ".join(
            ["%s" % items[0][1]] + ["%s=%s" % item for item in items[1:]]
        )

        return ret

    call.__name__ = constructor.__name__
    # call.__getargspec__ = getargspec(constructor)
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

    def play(decision, participants, observers=[], tremble=np.NaN):
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

        intention = decider.decide(decision, participant_ids)
        
        if np.random.rand() < decision.tremble:
            action = np.random.choice(decision.actions)
        else:
            action = intention

        payoffs = copy(decision(action))

        observer_ids = tuple(
            observer.world_id for observer in set(list(observers) + list(participants))
        )

        observations = [
            {
                "game": decision,
                "action": action,
                "participant_ids": array(participant_ids),
                "observer_ids": frozenset(observer_ids),
                "payoffs": array(payoffs),
            }
        ]

        # observations = [(decision,participant_ids,observer_ids,action)]
        return payoffs, observations, None

    def next_game(self):
        try:
            g = self.playable.next_game()
            self.N_players = g.N_players
            return self
        except:
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

    def __init__(self, observability, playable):
        self.name = "RandomlyObserved(%s,%s)" % (observability, playable.name)
        self.observability = observability
        self.N_players = playable.N_players
        self.playable = playable

    def play(self, participants, observers=[], tremble=np.NaN):
        if self.observability < 1:
            # Sample from the list of possible observers
            observers = np.random.choice(
                observers, size=int(len(observers) * self.observability), replace=False
            )
        elif self.observability == 0:
            observers = []

        payoffs, observations, notes = self.playable.play(
            participants, observers, tremble
        )

        for observer in set(list(observers) + list(participants)):
            # Check if the observer implements observe
            if hasattr(observer, "observe"):
                observer.observe(observations)

        return payoffs, observations, notes


class PrivatelyObserved(RandomlyObserved):
    def __init__(self, playable):
        super(PrivatelyObserved, self).__init__(0, playable)
        self.name = "PrivatelyObserved(%s)" % playable.name


class PubliclyObserved(RandomlyObserved):
    def __init__(self, playable):
        super(PubliclyObserved, self).__init__(1, playable)
        self.name = "PubliclyObserved(%s)" % playable.name


class AllNoneObserve(Playable):
    """
    randomly selects a specified percent of the provided observers
    percent is determined by self.observability, which must be set beforehand
    these observers are passed down into _play
    selected observers and all participants observe
    """

    def __init__(self, observability, playable, **kwargs):
        self.name = "AllNoneObserve(%s,%s)" % (observability, playable.name)
        self.observability = observability
        self.N_players = playable.N_players
        self.playable = playable

    def next_game(self):
        g = self.playable.next_game()
        self.N_players = g.N_players
        return self

    def play(self, participants, observers=[], tremble=np.NaN):
        if np.random.rand() < self.observability:
            observers = observers
        else:
            observers = []

        payoffs, observations, notes = self.playable.play(
            participants, observers, tremble
        )

        observers = frozenset(list(observers) + list(participants))

        id_to_observer = {observer.world_id: observer for observer in observers}
        observer_indices = frozenset(list(id_to_observer.keys()))

        for observer_index in observer_indices:
            if hasattr(id_to_observer[observer_index], "observe"):
                id_to_observer[observer_index].observe(observations)

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

    def __init__(self, payoffDict, tremble):
        actions = list(payoffDict.keys())
        self.N_players = len(list(payoffDict.values())[0])
        self.actions = actions
        self.action_lookup = dict(list(map(reversed, enumerate(actions))))
        self.payoffs = payoffDict
        self.tremble = tremble

    def __call__(self, action):
        self.last_action = action
        return array(self.payoffs[action])


def BinaryDictatorDict(endowment=0, cost=COST, benefit=BENEFIT):
    return {"keep": (endowment, 0), "give": (endowment - cost, benefit)}


def BinaryDictator(endowment=0, cost=COST, benefit=BENEFIT, tremble=np.NaN):
    """
    a 2-participant decision
    """
    decision = Decision(BinaryDictatorDict(endowment, cost, benefit), tremble)
    decision.tremble = tremble
    decision.name = decision._name = "BinaryDictator(%s)" % ",".join(
        map(str, [endowment, cost, benefit])
    )
    return decision


@literal
def SocialDictator(
    endowment=ENDOWMENT, cost=COST, benefit=BENEFIT, intervals=2, tremble=np.NaN, **kwargs
):
    cost = float(cost)
    benefit = float(benefit)
    max_d = benefit - cost
    max_r = cost / benefit

    # np.random.uniform(0, cost)

    ratios = np.linspace(0, max_r, intervals)
    differences = np.linspace(0, max_d, intervals)

    def new_cost(r, d):
        return d / (1 - r) - d

    def new_benefit(r, d):
        return d / (1 - r)

    payoffs = [
        (endowment - new_cost(r, d), new_benefit(r, d))
        for d, r in zip(differences, ratios)
    ]

    if intervals == 2:
        decision = Decision({"keep": payoffs[0], "give": payoffs[1]})
    else:
        decision = Decision(OrderedDict((str(p), p) for p in payoffs))

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
        self.play_ordering_pairs = [
            (decision, ordering) for decision, ordering in self.decision_ordering_pairs
        ]

    def matchups(self, participants):
        """
        this simply returns an iterator whose elements are the
        (decision.play,ordering) tuples that will be played

        note that the first element is a 'play' function and not the decision itself.

        this method is meant to be overwritten by child-classes
        """
        return iter(self.play_ordering_pairs)

    def play(self, participants, observers=[], tremble=np.NaN):
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
        # initialize accumulators
        observations = []
        payoffs = np.zeros(len(participants))

        for game, ordering in self.matchups(participants):
            pay, obs, rec = game.play(participants[ordering], observers, tremble)
            payoffs[ordering] += pay
            observations.extend(obs)

        return payoffs, observations, []

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


class SymmetricMatchup(object):
    """
    plays the game with every possible permutation of the participants


    NOTE:
    there will be redundancy if ordering does not matter.
    this is best for games where every position is different.
    """

    def __init__(self, game):
        self.game = game
        self.N_players = game.N_players
        self.cached_matchups = None
        self.n_participants = None

    def matchups(self, participants):
        # NOTE: This makes a key assumption that the number of participants will be constant every time this is called otherwise the cache will not be accurate.
        if self.cached_matchups is None:
            matchups = list(permutations(range(len(participants)), self.game.N_players))
            np.random.shuffle(matchups)
            self.cached_matchups = matchups
            self.n_participants = len(participants)

        # Make sure the number of particpants is the same as the initial call
        assert len(participants) == self.n_participants

        game = self.game
        for matchup in self.cached_matchups:
            yield (game, list(matchup))


class Symmetric(SymmetricMatchup, DecisionSeq):
    pass


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

    def annotate(self, participants, payoff, observations, record):
        raise NotImplementedError

    def play(self, participants, observers=None, tremble=np.NaN, notes={}):
        if observers is None:
            observers = participants

        # initialize accumulators
        observations = []
        payoffs = np.zeros(len(participants))
        record = [
            {
                "round": 0,
                "payoff": payoffs,
                "beliefs": tuple(
                    copy(getattr(agent, "belief", None)) for agent in participants
                ),
                "likelihoods": tuple(
                    copy(getattr(agent, "likelihood", None)) for agent in participants
                ),
                "new_likelihoods": tuple(
                    copy(getattr(agent, "new_likelihoods", None))
                    for agent in participants
                ),
            }
        ]

        record = []

        record.append(self.annotate(participants, payoffs, [], [], notes))

        for game, ordering in self.matchups(participants):
            pay, obs, rec = game.play(participants[ordering], observers, tremble)
            new_payoffs = np.zeros(len(participants))
            new_payoffs[ordering] += pay
            payoffs += new_payoffs
            record.append(self.annotate(participants, new_payoffs, obs, rec, notes))
            observations.extend(obs)

        assert len(payoffs) == len(participants)
        return payoffs, observations, record

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
        self.name = self._name = "Repeated(" + str(rounds) + "," + game.name + ")"
        self.game = game
        self.rounds = rounds
        self.N_players = game.N_players
        self.current_round = 0

    def annotate(self, participants, payoff, observations, record, notes):
        note = {
            "round": self.current_round,
            "actions": tuple(observation["action"] for observation in observations),
            "actors": tuple(
                observation["participant_ids"] for observation in observations
            ),
            "payoff": copy(payoff),
            "games": tuple(observation["game"] for observation in observations),
            "beliefs": tuple(
                copy(getattr(agent, "belief", None)) for agent in participants
            ),
            "likelihoods": tuple(
                deepcopy(getattr(agent, "likelihood", None)) for agent in participants
            ),
            "new_likelihoods": tuple(
                copy(getattr(agent, "new_likelihoods", None)) for agent in participants
            ),
        }
        note.update(notes)
        return note

    def matchups(self, participants):
        ordering = list(range(len(participants)))
        for game_round in range(1, self.rounds + 1):
            self.current_round = game_round
            yield self.game, ordering


class Annotated(AnnotatedDS):
    def __init__(self, rounds, game):
        self.rounds = rounds
        self.name = self._name = "Annotated(" + game.name + ")"
        self.game = game
        # self.N_players = game.N_players
        self.current_round = 0

    def annotate(self, participants, payoff, observations, record, notes):
        note = {
            "round": self.current_round,
            "actions": tuple(observation["action"] for observation in observations),
            "actors": tuple(
                observation["participant_id"] for observation in observations
            ),
            "payoff": copy(payoff),
            "games": tuple(observation["game"] for observation in observations),
            "beliefs": tuple(
                copy(getattr(agent, "belief", None)) for agent in participants
            ),
            "likelihoods": tuple(
                deepcopy(getattr(agent, "likelihood", None)) for agent in participants
            ),
            "new_likelihoods": tuple(
                copy(getattr(agent, "new_likelihoods", None)) for agent in participants
            ),
        }

        note.update(notes)
        return note

    def matchups(self, participants):
        for game_round, thing in zip(
            range(1, self.rounds + 1), cycle(self.game.matchups(participants))
        ):
            self.current_round = game_round
            yield thing


class AnnotatedGame(AnnotatedDS):
    """
    meant to replace AnnotatedDS
    treats every matchup produced by the wrapped game's "matchup" method
    as a "round"
    """

    def __init__(self, game):
        self.name = self._name = "AnnotatedGame(" + game.name + ")"
        self.game = game
        self.current_round = 0

    def play(self, participants, observers=None, tremble=np.NaN, notes={}):
        if observers is None:
            observers = participants

        # initialize accumulators
        observations = []
        payoffs = np.zeros(len(participants))
        record = []

        # record basic info for round 0
        record.append(self.annotate(participants, payoffs, [], [], notes))

        for game, ordering in self.matchups(participants):
            new_payoffs = np.zeros(len(participants))
            pay, obs, rec = game.play(participants[ordering], observers, tremble)
            new_payoffs[ordering] += pay
            payoffs += new_payoffs
            record.append(self.annotate(participants, new_payoffs, obs, rec, notes))
            observations.extend(obs)

        assert len(payoffs) == len(participants)
        return payoffs, observations, record

    _play = play

    def annotate(self, participants, payoff, observations, record, notes):
        note = {
            "round": self.current_round,
            "actions": tuple(observation["action"] for observation in observations),
            "actors": tuple(
                observation["participant_ids"] for observation in observations
            ),
            "payoff": copy(payoff),
            "games": tuple(observation["game"] for observation in observations),
            "beliefs": tuple(
                copy(getattr(agent, "belief", None)) for agent in participants
            ),
            "likelihoods": tuple(
                deepcopy(getattr(agent, "likelihood", None)) for agent in participants
            ),
            "new_likelihoods": tuple(
                copy(getattr(agent, "new_likelihoods", None)) for agent in participants
            ),
        }

        note.update(notes)
        return note

    def matchups(self, participants):
        for game_round, game_ordering_pair in zip(
            itertools.count(1), self.game.matchups(participants)
        ):
            self.current_round = game_round
            yield game_ordering_pair


class RandomizedMatchup(DecisionSeq):
    def __init__(self, rounds, game, deterministic=False, **kwargs):
        self.rounds = rounds
        self.game = game
        self.deterministic = deterministic

        if self.deterministic == True:
            assert isinstance(self.rounds, int)

        self.name = self._name = (
            "RandomizedMatchup(" + str(rounds) + ", " + game.name + ")"
        )

    def matchups(self, participants):
        player_count = len(participants)
        indices = np.arange(player_count)
        counts = np.zeros(shape=(player_count, player_count))

        if self.deterministic:
            for i, j in combinations(list(range(player_count)), 2):
                counts[i, j] = counts[j, i] = self.rounds
        else:
            for i, j in combinations(list(range(player_count)), 2):
                counts[i, j] = counts[j, i] = np.random.geometric(1.0 / self.rounds)

        # while there is anyone that has more than two partners remaining
        while True:
            game = self.game.next_game()
            N_players = game.N_players
            N_participants = N_players - 1

            # for each decider count the number of possible others
            # they can still interact with.
            sums = (counts > 0).sum(axis=1)

            # only pick from those that have sufficient possible
            # interaction partners.
            decider_pool = indices[sums >= N_participants]

            if len(decider_pool) == 0:
                break

            decider = np.random.choice(decider_pool)

            recipient_pool = indices[counts[decider] > 0]

            recipients = np.random.choice(recipient_pool, N_participants, replace=False)

            for recipient in recipients:
                counts[decider, recipient] -= 1
                counts[recipient, decider] -= 1

            assert counts.min() >= 0
            yield (game, [decider] + list(recipients))


class Dynamic(Playable):
    """
    takes a playable-making function
    and a thunk that generates parameters for it
    whenever it's 'play' method is called, it generates
    new arguments and serves up a playable made with them
    """

    def __init__(self, playable_constructor):
        self.constructor = playable_constructor
        self.next_game()

    def new(self):
        return self.constructor(**self.arg_gen())

    def next_game(self):
        # import pdb; pdb.set_trace()
        g = self.current_game = self.constructor()
        # try:
        # except TypeError as e:
        #     g = self.current_game = apply(self.constructor,self.arg_gen())

        self.N_players = g.N_players
        return g

    def play(self, participants, observers=[], tremble=np.NaN):
        to_play = self.current_game.play
        self.next_game()
        return to_play(participants, observers, tremble)


@literal
def RepeatedPrisonersTournament(
    rounds, cost, benefit, tremble, **kwargs
):
    PD = Symmetric(BinaryDictator(cost=cost, benefit=benefit, tremble=tremble))
    g = Repeated(rounds, PrivatelyObserved(PD))

    return g

@literal
def RepeatedSequentialPrisonersDilemma(
    rounds, cost, benefit, tremble, **kwargs
):
    PD = Symmetric(PrivatelyObserved(BinaryDictator(cost=cost, benefit=benefit, tremble=tremble)))
    g = Repeated(rounds, PD)

    return g

direct_seq = RepeatedSequentialPrisonersDilemma
direct = RepeatedPrisonersTournament


def engine_gen(nactions, max_players, benefit, cost, tremble):
    # Number of actions, not including the zero-action.
    N_actions = 1 + np.random.poisson(nactions - 1)

    # Number of players that will be affected by the decision
    N_players = np.random.choice(list(range(2, max_players + 1)))

    # initialize set of choices with the zero-action. All games
    # contain the option to "do nothing"
    choices = [np.zeros(N_players)]

    for n in range(N_actions):
        c = np.random.poisson(cost)
        b = np.random.exponential(benefit - cost)
        for p in range(1, N_players):
            choice = np.zeros(N_players)
            choice[0] = -c
            choice[p] = c + b
            assert -choice[0] < choice[p]

            choices.append(copy(choice))

    decision = Decision(OrderedDict((i, p) for i, p in enumerate(choices)), tremble)
    decision.tremble = tremble

    return decision


@literal
def game_engine(
    rounds,
    observability,
    cost=1,
    benefit=10,
    nactions=2,
    tremble=np.NaN,
    max_players=3,
    **kwargs
):
    assert nactions >= 0

    dictator = Dynamic(
        lambda: engine_gen(nactions, max_players, benefit, cost, tremble)
    )
    dictator.name = "dynamic"
    game = AnnotatedGame(
        RandomizedMatchup(
            rounds, AllNoneObserve(observability, dictator, **kwargs), **kwargs
        )
    )
    return game


if __name__ == "__main__":
    pass
