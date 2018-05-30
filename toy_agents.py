from itertools import chain, combinations
from copy import copy
import numpy as np
from collections import defaultdict

from utils import normalized, softmax

def powerset(s):
    s = list(s)
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

def subsets_with(base_set, common):
    assert  common <= base_set
    subsets = powerset(base_set-common)
    return map(common.union, subsets)

def add_tremble(p, tremble):
    if tremble == 0:
        return p
    else:
        return (1 - tremble) * p + tremble * np.ones(len(p)) / len(p)

class Agent(object):
    """
    Agents that have these methods and exactly these parameters can be modeled by RationalAgents
    only 'act' and 'observe' should ever be called by the World
    """
    def act(self, game, participant_ids):
        """this method is the only way 'Agents' affect the world"""
        action_likelihoods = self.likelihood_of(game, participant_ids)
        return np.random.choice(game.actions, p = action_likelihoods)

    def likelihood_of(self, game, participants, tremble = 0, action = None):
        """this method is semi-private and meant to be used internally by the agent or by other agents
        that contain an instance of it"""
        pass

    def observe(self, simultaneous_observations):
        """this method is the only way the internal state of an 'Agent' should be changed"""
        pass

class model_node(object):
    def __init__(self, agent_ids, common_knowledge):
        agent_types = common_knowledge['agent_types']
        agent_ids = common_knowledge['agent_ids']

        self.model = {}
        self.belief = {}
        self.likelihood = {}

        for agent_id in agent_ids:
            self.belief[agent_id] = copy(common_knowledge['prior'])
            self.likelihood[agent_id] = np.zeros(len(agent_types))

            # self.model and the code below is only necessary for
            # other models that implement observe in a different way
            # i.e., anything that is not a subclass of RationalAgent
            self.model[agent_id] = {}
            for agent_type in agent_types:
                if not issubclass(agent_type, RationalAgent):
                    self.model[agent_id][agent_type] = agent_type(agent_id, common_knowledge)
class FullLattice(dict):
    def __init__(self, agent_set, common_knowledge):
        agent_ids = common_knowledge['agent_ids']
        for subset in subsets_with(agent_ids, agent_set):
            lattice[subset] = model_node(subset, common_knowledge)

class UniverseSet(frozenset):
    def __contains__(self,x):
        return True
    def __and__(self,s):
        return s
    def __rand__(self,s):
        return s
    def __or__(self,s):
        return self
    def __ror__(self,s):
        return self
    def __len__(self):
        # TODO: This needs to be infinity but using approx infinity
        # for now due to some other bug somewhere
        return 10000000
    def __le__(self,other):
        return False
    def __lt__(self,other):
        return False
    def __ge__(self,other):
        return True
    def __gt__(self,other):
        return True
    def issuperset(self,other):
        return True
    def issubset(self,other):
        return False
    def __rsub__(self,other):
        return other-other
Universe = UniverseSet()

class ObserverLattice(object):
    class ModelDict(dict):
        def __init__(self, common_knowledge):
            self.model_types = [t for t in common_knowledge['agent_types'] if not issubclass(t, RationalAgent)]
        def __missing__(self, agent_id):
            model_dict = {model_type : model_type(agent_id, common_knowledge) for model_type in self.model_types}
            self[agent_id] = model_dict
            return model_dict

    class JointState(object):
        def __init__(self, common_knowledge):
            agent_types = common_knowledge['agent_types']
            prior = common_knowledge['prior']
            self.belief = defaultdict(lambda:copy(prior))
            self.likelihood = defaultdict(lambda:np.zeros_like(prior))
            self.model = ModelDict(common_knowledge)

    def __init__(self, common_knowledge):
        self[Universe] = JointState(common_knowledge)

    def insert(self, new_set):
        known_sets = list(self.keys())
        new_sets = defaultdict(list)

        for known_set in known_sets
            intersect = known_set&new_set
            if intersect not in known_sets:
                new_sets[intersect].append(known_set)

        for new_set, supersets in new_sets:
            smallest_superset = sorted(supersets, key = len)[0]
            self[new_set] = deepcopy(self[smallest_superset])

    def __missing__(self, agent_set):
        """
        when asked for the joint state of an uknown set,
        return the state of the smallest superset of the given set
        because this is a join-semilattice, there is guaranteed to be a least upper bound
        """
        smallest_superset = sorted(filter(lambda s: s > agent_set, self.keys()), key = len)[0]
        return self[smallest_superset]

class RationalAgent(Agent):
    """
    This subclasses of RationalAgent can simulate Agent sublcasses and other RationalAgent subclasses
    Rational subclasses only require a 'utility' method to be defined
    """
    def __init__(self, agent_id, common_knowledge):
        self.agent_id = agent_id
        agent_set = self.agent_set = frozenset([agent_id])

        self.common_knowledge = common_knowledge
        self.beta = common_knowledge['beta']
        agent_ids = common_knowledge['agent_ids']

        lattice = self.lattice = FullLattice(agent_set, common_knowledge)

        me = lattice[self.agent_set]
        self.belief = me.belief
        self.likelihood = me.likelihood

    def act(self, game, participant_ids):
        # TODO: Check if it should just be self.likelihood_of : https://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python
        action_likelihoods = self.__class__.likelihood_of(game, participant_ids, self.belief, self.common_knowledge)
        return np.random.choice(game.actions, p = action_likelihoods)

    def observe(self, simultaneous_observations):
        agent_set = self.agent_set
        agent_types = self.common_knowledge['agent_types']

        # this is a map of type set(id)->id->array
        # it keeps track of the likelihoods that need updating for which subsets of observers
        new_likelihoods = defaultdict(lambda: defaultdict(lambda: np.zeros_like(self.common_knowledge['prior'])))

        for observation in simultaneous_observations:
            #actor is in participants whose elements are a subset of observers
            participants, observers = [observation[attr] for attr in ['participant_ids','observer_ids']]
            actor = participants[0]

            for subset in subsets_with(observers, frozenset([self.agent_id, actor]):
                joint = self.lattice[subset]

                type_likelihoods = []
                for potential_type in agent_types:
                    if issubclass(potential_type, RationalAgent):
                        likelihood = potential_type.likelihood_of(belief = joint.belief,
                                                                  common_knowledge = self.common_knowledge,
                                                                  **observation)
                    else:
                        # Get the likelihood of the non-rational types. 
                        likelihood = joint.model[actor][potential_type].likelihood_of(**observation)
                        
                    type_likelihoods.append(likelihood)

                new_likelihoods[subset][actor] += np.log(type_likelihoods)


        # update lattice beliefs using new_likelihoods
        for observers, actors_likelihood in new_likelihoods.iteritems():
            joint = self.lattice[observers]
                                      
            for actor, new_likelihood in actors_likelihood.iteritems():
                joint.likelihood[actor] += new_likelihood
                joint.belief[actor] = normalized(np.exp(common_knowledge['prior']+joint.likelihood[actor]))

            # Update the non-rational types
            for agent_type in agent_types:
                if not issubclass(agent_type, RationalAgent):
                    for observer in observers:
                        joint.model[observer][agent_type].observe(simultaneous_observations)

    @classmethod
    def likelihood_of(cls, game, participant_ids, belief, common_knowledge, tremble = 0, action = None, **kwargs):
        """
        this is a class method so that 'RationalAgents' can model each other without needing to
        build entire copies of each other.
        Here we assume that all 'RationalAgents' share the same theory of mind and priors.
        """
        utility_per_action = [cls.utility(game.payoffs[a], participant_ids, belief, common_knowledge) for a in game.actions]
        likelihoods = add_tremble(softmax(utility_per_action, common_knowledge['beta']), tremble)

        if action == None:
            return likelihoods
        else:
            return likelihoods[game.actions.index(action)]

    @classmethod
    def utility(cls, payoffs, participant_ids, belief, common_knowledge):
        """
        This is a class method for the same reason 'likelihood_of' is
        """
        raise Warning("RationalAgent is a non-functional ABC. Create a subclass with 'utility' method defined")

### The Rational Agent of interest
class ReciprocalAgent(RationalAgent):
    """
    ReciprocalAgent cares about an agent's payoffs proportional to its belief that they are
    also ReciprocalAgents
    """
    @classmethod
    def utility(cls, payoffs, participant_ids, belief, common_knowledge):
        cls_index = common_knowledge['agent_types'].index(cls)
        weights = [belief[p_id][cls_index] for p_id in participant_ids]
        weights[0] = 1
        return np.dot(payoffs,weights)

class SelfishAgent(RationalAgent):
    @classmethod
    def utility(cls, payoffs, participant_ids, belief, common_knowledge):
        weights = [0 for p_id in participant_ids]
        weights[0] = 1
        return np.dot(payoffs,weights)

class TFT(Agent):
    def __init__(self, agent_id, *args, **kwargs):
        self.agent_id = agent_id
        self.last_of = defaultdict(lambda:"give")

    def observe(self, PD_observations):
        agent_id = self.agent_id
        [A,B] = PD_observations
        [alice, bob] = participants = A['participant_ids']
        assert [bob, alice] == B['participant_ids']

        if agent_id == alice:
            self.last_of[bob] = B['action']
        if agent_id == bob:
            self.last_of[alice] = A['action']

    def likelihood_of(self, game, participant_ids, tremble = 0, action = None, **kwargs):
        my_id, opponent_id = participant_ids
        last_action = self.last_of[opponent_id]
        likelihoods =  add_tremble([1 if a == last_action else 0 for a in game.actions], tremble)
        if action == None:
            return likelihoods
        else:
            return likelihoods[game.actions.index(action)]

class Playable(object):
    def play(self, participants, observers, tremble):
        """
        'participants' is a list of agents (not agent_ids)
        'observers' is a list of agents
        'tremble' is a scalar between 0 and 1

        this method returns an array-like of the same length as 'participants'
        where each entry corresponds to the payoff of the agent in the
        corresponding place in the 'participants' list.

        this method is expected to call the 'act' method of Agents
        by convention the first agent in 'participants' is the decider
        this method is meant to be called by the world.
        """
        pass

def observation(game, participant_ids, observer_ids, action, tremble):
    return locals()

common_knowledge = dict(agent_ids = frozenset("ABCD"), agent_types = [ReciprocalAgent, SelfishAgent, TFT], prior = [.33,.33,.33], beta = 5)

a = ReciprocalAgent("A", common_knowledge)
#b = a.lattice[frozenset([0,1])]
#print a.agent_id
print a.__dict__.keys()
#print b.__dict__.keys()
#print a.belief[0]

from games import BinaryDictator
bd = BinaryDictator(cost = 1, benefit = 10)


def observation(participant_ids, action, game = bd, observer_ids = None, tremble = 0):
    if observer_ids == None:
        observer_ids = participant_ids
    return dict(locals(),**dict(observer_ids = frozenset(observer_ids)))

def PD_obs(actions, participants = "AB", tremble = 0):
    return [observation(participants, actions[0], bd, participants, tremble),
            observation(list(reversed(participants)), actions[1], bd, participants, tremble)]

#obs = [[observation("AB",'keep','AB')],[observation('BA','keep','ABC')]]

tft_obs = [PD_obs(["keep","give"]), PD_obs(["give","keep"]), PD_obs(["give","give"])]

for o in tft_obs:
    a.observe(o)
    #print a.belief['A']
    print a.belief['B']


#print ReciprocalAgent.likelihood_of(belief = a.belief, common_knowledge = common_knowledge, **obs[0])
#print SelfishAgent.likelihood_of(belief = a.belief, common_knowledge = common_knowledge, **obs[0])-
#print "A.A",a.belief['A']
#print "A defects B"
#a.observe(obs[0])
#print "A.A", a.belief['A']
#print "A.B", a.belief['B']
#print "B defects A"
#a.observe(obs[1])
#print "A.B", a.belief['B']
#print "AC.A", a.lattice[frozenset('AC')].belief['A']
#print "AC.B", a.lattice[frozenset('AC')].belief['B']
