# June 12

1. Adding support for many types of agents (the altruistic type). This will require updating the code in the observe_k function as well as writing a version of sample\_alpha that is more abstract.

1. Adding more complicated games, including multiple games, including sequential games, simultaneous games (that have limited observability). The ultimate version of this is to build a "grammar over games". Stringing games together etc.

1. Implement learning about the person "acted upon". If person 3 knows person 1 is nice (from previous interactions) and person 3 doesn't know anything about person 2 (has never seen him act). If person 1 is mean to person 2, person 3 should infer that person 1 knew something about person 2 even though person 3 didn't see anything. Try capturing this by allowing the prior agents have over other agents to be a random variable and hence learnable. Think about the connection between the probability that an interaction is observed and the parameter of how likely other agents are to have a known value.

1. Also think about doing Gibbs sampling to check our model against (which is not online). 

# Jan 26

TODO:
We should split up everything into Agents, World, Games, Experiments and Utils.
Information pertinent to the agents vs the world should be separated accordingly.
Make clean params and genome creators based on this.
Port all experiments into the new format, using multi_call

default_params update docstring stop conditions
multi_call hash args in call without the trial and use it as a field when storing.
Work out the mechanisms of observability vs observation.
Try to figure out a way to decouple observation from observation.
Remember that we might want observers known at decision time as it might affect choices.

1. Agents (indirect_reciprocity.py)

2. Puppet: This kind of agent prompts a user for its decisions every time it plays

2. SelfishAgent: Maximizes its own payoff

2. AltruisticAgent: Maximizes the sum of payoffs

2. RationalAgent: Base class. Inheriting from this class gives an agent the model, likelihood, and belief attributes and methods for initializing and updating these, most importantly the observe and observe_k methods.

2. IngroupAgent: Base class. Those who inherit from this class should define an ingroup method, which returns a list of types. IngroupAgeents value an agent's payoff proportional to the belief that the agent in question is of a type in this list.

2. RecpirocalAgent: An IngroupAgent that cares about only its own type.

2. Nice ReciprocalAgent: An IngroupAgent that cares about its own type and AltruisticAgents.

1. Utility functions (indirect_reciprocity.py)

2. default_params: returns a dictionary with common parameters used by agents and World. the function will overwrite any of the parameters in this dict if provided with them as keyword arguments in the function call. Most common use is to define a dict with parameters one cares about and feed it to this function to fill in the rest, (eg. "params = default_params(**dict)") 

2. generate_random_genomes: takes N_agents, agent_types_world, agent_types among others. makes N_agents number of agents. each of the possible types is sampled from agent_types_world, but each agent thinks only agents in agent_types can exist.

2. prior_generator(agent_types,RA_prior): returns an np array where entries correspond to a prior over an agent being of a given type. indices correspond to a type's position in agent_types. if priors are explicitly given in RA_prior (which is a dict of type Agent->Number) then those priors are set accordingly; types not listed are given a uniform prior over the remaining probability.

2. default_genome(params,agent_type = False): makes a standard genome with given params. if no agent_type is given it chooses randomly from agent_types in params.

2. generate_proportional_genomes(params,agent_proportions): returns approximately 'N_agents' number of agents, as specified in 'params'. if 'agent_proportions' defaults, it makes exactly 'N_agents' randomly. agent_proportions is expected to be Agent->Fraction, but it's not enforced. otherwise it expects that agent_proportions tells you what fraction of the population will be of each type. it rounds up fractions. THIS FUNCTION SHOULD BE AMENDED TO GUARANTEE THAT THE PROPORTIONS ARE HONORED, DESPITE THE POPULATION SIZE.

2. World(params,genomes): Right now this just makes agents out of genomes and plugs them into the world. It will, later on, handle changes to the population.

1. Experiments

2. multi_call decorator(experiment_utils.py): Designed as a way to succinctly run an experiment with many different combinations of parameters. This decorator is to be used with functions meant to be run with many parameters. If a decorated function is called with any parameters as a lists, it is taken to mean that we want that parameter to take on each of those values in turn. The function will be called multiple times with the values of the constant parameters being fixed, but with the variable parameters taking on each of their possible value combinations.

2. fitness_v_selfish: returns the ratio of the average payoffs of a population of a given AgentType vs SelfishAgents. Specified are the population size and what percent of it is agent_type.

1. games.py: defines all the decisions and games agents can make. games are matchmaking engines that composable in different ways.

2. Playable: the primary base class, defines the 'play' method which takes in a decision and participants, implicitly specifying that a decision is a dict with named actions and as many payoffs as participants. CONSIDER MERGING WITH DECISION, NOT GETTING MILEAGE OUT OF INHERITANCE.

2.Decision: Given the correct number of agents, has the first agent make a decision, returns a tuple of the payoffs, the observation event, and a None. This is the only thing agents actually know how to interact with. Every other class in games.py orchestrates different ways to get a pool of agents to actually play out a decision. Initialize it with a payoff dict and play with the 'play' method. ADD INPUT VALIDATION TO MAKE SURE NUMBER OF PLAYERS IS CORRECT. 

2. games/matchups: from this point on, classes are ways to arrange multiple decisions played out in sequence. these consist of two parts, 'play' which actually calls the underlying playable, and 'matchups' which generates the decision/order pairings to be played. 

2. DecisionSeq: Functionally a base class, most importantly defines 'play' behavior for games with 'matchups'. This class captures explicitly specified matchups, other classes inherit from this one and act as wrappers for specifying the matchups. One could also manually specify these, but it's unadvisable. It is initialized with a list of playable/agent_indices pairs. When DecisionSeq.play(agents) is called, it goes through this list and calls the decision.play method of the corresponding decision with the agents specified in the agent_indices list. If the first index is i, then the ith agent in the 'agents' list will be first, and so on. The 'agents' list is actually expected to be a numpy array, and is actually sliced using a call of the form <agents[agent_indices]>.
MATCHUPS HERE RETURNS GAME.PLAY/ORDERING PAIRS AS OPPOSED TO GAME/ORDERING, IS THIS RIGHT? MIGHT HAVE GONE UNNOTICED BECAUSE WE NEVER REALLY USE IT DIRECTLY.

2. CombinatorialMatchup: Base Class, overwrites 'matchups' and 'init'. Used to define a game where you play the same subgame with every combination (not permutation) of given participants, this is best used for subgames that require a specific number of players and handle their own ordering. The order in which each combination is served is randomized. The game adopts the number of players K as stored in the subgame, but its 'play' method can be called with any number of players N greater or equal to K and it will still work. In this case it will make the N choose K number of matchups. In the case where N == K, it is exactly the same as calling the subgame directly. Initialized with just a playable.

2. Combinatorial: Inherits from CombinatorialMatchup and DecisionSeq to combine 'matchups' from the first and 'play' from the second. This is what is used for independent games (i.e. those in which one game does not depend on the last).

2. SymmetricMatchup: Same as CombinatorialMatchup, except that this plays all permutations of players given. This is best used for games that don't handle their own ordering; good for playing all matchups for a decision. Prisoner's dilemma is easily defined as Symmetric(BinaryDictator()).

2. Symmetric: Same as combinatorial.

2. EveryoneDecidesMatchup: use this when the decider matters and all payoffs to non-deciders are symmetrical and of equal cost to the decider

2. EveryoneDecides: Same deal as Combinatorial, and Symmetric.

2. Dependent Games: Games which can't be expressed as a sequence of independent games/decisions. This implies that the events of one game can affect the next game that is played in some way. This usually also implies that each game is observed.

3. DecisionDependent: This is a declaration of a kind of decision whose payoffs are a function of another payoff. 

3. DecisionDependentSeq: This defines a new 'play' method where each subplayable is fed the payoffs of the previous before being played out.

3. UltimatumGame: defined by it's endowment alone. it is a combination of UltimatumPropose and UltimatumDecide.

3. UltimatumPropose: specified by an endowment, presents every option to give or keep every possible 2 integer split of the endowment.

3. UltimatumDecide: specified by a proposed split. Accept is a 0 vector, Reject is the negative of the proposal.

2. AnnotatedDS: defines an 'annotate' method and a 'play' method that calls 'annotate' after each play of the subgame. this is used to create a history of play.

2. Repeated: specifies that a game is to be played some number of times, records the current round, a copy of the players, observations, payoffs,beliefs, and likelihoods after the round itself. the states of players, their beliefs and likelihoods are incremental between rounds.

2. RepeatedPrisonersTournament: specify with number of rounds. Given any number of agents, plays every ordered pair against each other once. All interactions are observed at the end. Then this is repeated some number of times, as specified by the initialization.

2. Observation wrappers: Publicly-,Privately-, and Randomly-...-Observed overwrite the play function and pass down no, all, or a random subset of observers. 