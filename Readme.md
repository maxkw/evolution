# TODO
## General
- delete old unused files (keep local copy, just in case)
- make experiment-specific documentation 'Experiments.md'
- Explain the what, how, and how-to for each existing experiment
- Make correctly-labeled plots for existing experiments

## Exploring 1v1
Make some plots for `binary_matchup`
1. What do these distributions look like
1. How many samples do we need to get it to converge (limit number of trials as a metric)
1. Figure out how much time it takes to generate that many samples
## indirect_reciprocity.py
1. We should split up everything into Agents, World, Games, Experiments and Utils.
1. Information pertinent to the agents vs the world should be separated accordingly.
1. Make clean params and genome creators based on this. 
1. `default_params` update docstring stop conditions
1. Refactor the way observability works. Work out the mechanisms of observability vs observation. Try to figure out a way to decouple observation from observation. Remember that we might want observers known at decision time as it might affect choices.
## games.py
- Consider merging `Playable` with `Decision`, not getting mileage out of inheritance.
- Add input validation to `Decision` make sure number of players is correct.
- Maybe make a single 'Observed' wrapper and many '...Observable' ones, and some shorthands.

# Docs

## Agents (indirect_reciprocity.py)

2. `Puppet`: This kind of agent prompts a user for its decisions every time it plays

2. `SelfishAgent`: Maximizes its own payoff

2. `AltruisticAgent`: Maximizes the sum of payoffs

2. `RationalAgent`: Base class. Inheriting from this class gives an agent the model, likelihood, and belief attributes and methods for initializing and updating these, most importantly the observe and observe_k methods.

2. `IngroupAgent`: Base class. Those who inherit from this class should define an ingroup method, which returns a list of types. IngroupAgeents value an agent's payoff proportional to the belief that the agent in question is of a type in this list.

2. `RecpirocalAgent`: An IngroupAgent that cares about only its own type.

2. `NiceReciprocalAgent`: An IngroupAgent that cares about its own type and AltruisticAgents.

<!-- ## Util functions (indirect_reciprocity.py) -->

<!-- 2. `default_params`: returns a dictionary with common parameters used by agents and World. the function will overwrite any of the parameters in this dict if provided with them as keyword arguments in the function call. Most common use is to define a dict with parameters one cares about and feed it to this function to fill in the rest, (eg. "params = default_params(**dict)")  -->

<!-- 2. `generate_random_genomes`: takes N_agents, agent_types_world, agent_types among others. makes N_agents number of agents. each of the possible types is sampled from agent_types_world, but each agent thinks only agents in agent_types can exist. -->

<!-- 2. `prior_generator(agent_types,RA_prior)`: returns an np array where entries correspond to a prior over an agent being of a given type. indices correspond to a type's position in agent_types. if priors are explicitly given in RA_prior (which is a dict of type Agent->Number) then those priors are set accordingly; types not listed are given a uniform prior over the remaining probability. -->

<!-- 2. `default_genome(params,agent_type = False)`: makes a standard genome with given params. if no agent_type is given it chooses randomly from agent_types in params. -->

<!-- 2. `generate_proportional_genomes(params,agent_proportions)`: returns approximately 'N_agents' number of agents, as specified in 'params'. if 'agent_proportions' defaults, it makes exactly 'N_agents' randomly. agent_proportions is expected to be Agent->Fraction, but it's not enforced. otherwise it expects that agent_proportions tells you what fraction of the population will be of each type. it rounds up fractions. THIS FUNCTION SHOULD BE AMENDED TO GUARANTEE THAT THE PROPORTIONS ARE HONORED, DESPITE THE POPULATION SIZE. -->

<!-- 2. World(params,genomes): Right now this just makes agents out of genomes and plugs them into the world. It will, later on, handle changes to the population. -->

## Experiments ##

<!-- 2. multi_call decorator(experiment_utils.py): Designed as a way to succinctly run an experiment with many different combinations of parameters. This decorator is to be used with functions meant to be run with many parameters. If a decorated function is called with any parameters as a lists, it is taken to mean that we want that parameter to take on each of those values in turn. The function will be called multiple times with the values of the constant parameters being fixed, but with the variable parameters taking on each of their possible value combinations. -->

1. The nature of experiments: When we talk about 'experiments' we refer to running a stochastic model with certain parameters a number of times per parameter set, and recording something about the state of the world after the model has run its course. In our case, we make a world, populate it with agents, and have them play a game, in which the agents will make decisions according to their nature and beliefs, and will be given rewards as a result of theirs and other agents' decisions. We can observe the results of the game and the state of agent's beliefs after a game has run its course and begin to understand how the initial conditions of the world relate to its final state.

1. Experimental Conditions: Our experiments take the form of functions, which have as arguments parameters to be fed into the various functions that specify the conditions of the model. Specifically our experiments mostly consist of creating some number of `Agents` and having them play a `game` using a `World` object to relate the two and actually run and record playthroughs. The parameters that specify the initial states of agents and the world take the form of well-defined dictionaries, for which we provide convenient constructor functions. The world is defined by parameters which are generated using the `default_params` function and by a list of agent genomes. Agents are completely specified by their genomes, these can be created individually with the `default_genome` function, or in batches using the functions `generate_random_genomes` and `generate_proportional_genomes`'. Games are specified differently and must be imported from `games.py`. Both the `default_...` functions are themselves basicaly pre-populated dictionaries whose keys can be overwritten by specifying a value in the function call (eg.`fun(a_key=a_val)`)

1. Using python idioms for fun and profit: Using dictionary unpacking and the `locals()` function, we can succinctly specify any of the parameters that affect our world.  Python allows function calls to be made by unpacking dictionaries, using the `**dict` idiom. That is to say: for some function defined as  `def fun(a,b=0,c=3,**kwargs):...` the statement `foo(**{'a':1,'b':2,'d':4})` is the same as `foo(a=1,b=2,c=3,rest={'c':3,'d':2'})`. Note that provided values overwrite their counterparts in the function definition, default values are overwritten if otherwise specified and used if not, and any unexpected arguments are stuffed into the `kwargs` variable. This works very well in conjunction with the built-in `locals()` function, which returns a dictionary of all defined variables in a scope thusfar. At the top of a function body, it conveniently captures all of the arguments passed in. Let's say we want to make an experiment in which our world contains two agent types, for example `ReciprocalAgent` and `SelfishAgent`, then we can write a function `def experiment1(agent_types=[ReciprocalAgent,SelfishAgent]):...` where when we want to define parameters to pass into `World`, we simply write `params = default_params(**locals())` and our variable `agent_types` will overwrite the default value for the function, while all other defaults are used. While this may seem a bit obtuse for the marginal gains, the benefit comes from the fact that if we want to specify another parameter, we only have to add that parameter as an argument in our experiment function's definition. This convention allows our experiment functions to remain succinct and only explicitly describe the more interesting differences between our experiments, other than just feeding in some different parameters.

1. Multi-call decorator: Using our decorator `multi-call()`, we can succinctly run a single experiment under many different conditionsand cache all the results to disk. In an experiment we usually want to run some specific combination of types and number of agents and kinds of games with many different starting conditions. Instead of handling this by running through all of these parameters in the function body each explicitly, each experiment describes only a single arrangement of the world, and the variation in the condition is specified by the function arguments. Our `multi-call()` decorator interprets any arguments that are fed in as lists as a range of conditions that we want our experiments to run, returning a pandas DataFrame whose columns are named for the arguments in the function plus a 'trial', 'result', and 'arghash' column. For example let's imagine we want to define a subtraction function `sub(a,b)` and we decorate it with `multi-call()` if we want to know the subtraction of b from a for every value of a and b from one to ten, we can simply write `sub(a=range(1,10),b=range(1,10))`, the decorator will expand this into the individual calls `sub(1,1),sub(1,2),...` and so on. Specifically it will run all combinations of values of arguments fed in as lists, any elements not fed in as lists will remain the same in each call. If one wishes to have an argument that needs to be a list and not be expanded one could simply write `fun(a=[[1,2,3]],b=3])` which would simply expand the list into the nested list and feed it into the function. Or one could use the decorator keyword `static` to ignore that argument, in the case above, one should decorate `fun` with the decorator call `multi-call(static=['a'])`, then the function call could simply be written as `fun(a=[1,2,3],b=3)`. If we want multi-call to ignore the order of some list parameters for the purposes of caching, like for example we want `agent_types=[[SelfishAgent,ReciprocalAgent]]` to be considered the same as `agent_types=[[ReciprocalAgent,SelfishAgent]]` in a function call, then that function should be decorated with `multi-call(unordered=['agent_type'])`. These keywords take in lists of argument names that exhibit special behaviors. 

1. Putting it all together: Check out the `binary_matchup` definition in `experiments.py` as an example of a simple two-type matchup and see how all of these instructions come together. Below are other experiments we provide.

fitness_v_selfish: returns the ratio of the average payoffs of a population of a given AgentType vs SelfishAgents. Specified are the population size and what percent of it is agent_type.

first_impression: Observer sees an agent cooperate a number of times before defecting, what does the observer think about them?

## Creating Games (games.py)
Defines all the decisions and games agents can make. games are matchmaking engines that are composable in different ways.

1. `playable`: for the purposes of this code a `playable` refers to any object that can be "played" by agents, i.e. any object that has a `play` method, the simplest of which is a `Decision`. The `games.py` script defines functions that are closed over playables, composing decisions into arbitrarily complicated multi-agent games. If constructed out of a subset of the provided tools, playables have a `.name` attribute that serves as a literal representation of how the playable is built. Such that eval(playable.name) will generate an identical copy of playable, unless the playable has been modified in some other way. 

1. `Decision`: A decision can be entirely specified by a simple dictionary which assigns names to payoff. The names can be any valid key (agents don't care, but you might), but the values must be sequences of numbers of some fixed size. This size defines the number of players in a decision, the order defines who gets which payoff. For example, we can define `binaryDictator = Decision({"keep":(1,0),"give":(0,2)})`. When we run the decision with `binaryDictator.play([Ale,Max])` then the first player, in this case `Ale`, is the decider. If they choose to `give` then they are giving themselves a `0` and `Max` a `2` payoff. The play function returns a triple consisting of (1) the payoff, (2) a list containing a single observation, which consists of the (2.1) decision, (2.2) a list of the player ids, a (2.3) list of observers, and (2.4) the name of the choice of the decider, and (3) a `None`. This last one is for compatibility purposes. We provide a BinaryDictator function that creates this decision and names itself appropriately.

1. Taking Turns: Many (all?) matrix games can be broken down into sequences of decisions. The Prisoner's Dilemma is simply two iterations of `BinaryDictator` where players trade places and everyone sees everyone else's choices after everyone has decided. We can capture this idea of players trading places with the `Symmetric` class, it takes a playable and makes a symmetric version of it, where the `decision` will be played with every *permutation* of the players. We can define Prisoner's Dilemma in a natural fashion as `PrisonersDilemma = Symmetric(BinaryDictator())`, we define a pretty maker function for Prisoner's Dilemma. When calling it with our agents as before, we now recieve (1) the cumulative payoff from both decisions, (2) a list of observations, and (3) an empty list instead of a `None`. The `Combinatorial` class makes games where the base game is played with every *combination* of the players, this is best used for base games that handle their own ordering or in which order doesn't matter. `EveryoneDecides` plays the game with every **meaningful combination** with the assumption that only the decider matters, and that the order of the remaining 'passive' players doesn't matter. The order in which the matchups are played is randomized.

1. Multiple Players/Tournaments: We can imagine that we might want three players to take turns playing this two-player game and tally up those results. The previously mentioned match-making classes will do exactly this when called with more than the number of players required for their base games.

1. Equivalent Games: The games `PD1 = Symmetric(BinaryDictator())` and `PD2 = Combinatorial(PrisonersDilemma()) = Combinatorial(PD1)` are functionally the same. In the first, we have every possible 2-player ordering of the given players play the `BinaryDictator` *decision*. `PD1` is considered a 2-player game by `Combinatorial`, in which 2 decisions are made, so `Combinatorial` pairs off the given player pool into all possible *subsets* of 2 players, who then go on to take turns when playing PD. The same number of decisions are made, the only thing that really changes in this case is the order in which the decisions are made.

1. Observation: Observation modifiers feed observations to `observers` passed into the appropriate field when the playable's 'play' method is called. Classes ending in `Observed` have this effect. `PrivatelyObserved` ignores any agents in the `observer` field, and has only the players observe. `PubliclyObserved` makes all agents in `observer` observe. `RandomlyObservable` is different, it selects a random subset of observers to be actually passed into the `observer` field, but does not have them observe immediately afterwards, the observation must be forced with 'PubliclyObserved'.

1. Simultaneous vs. Sequential: Observations change our games in subtle but profound ways by changing the information available to players. The games `Symmetric(PrivatelyObserved(BinaryDictator()))` and `Combinatorial(PrivatelyObserved(PrisonersDilemma))` are not the same. In the first, both players observe immediately after every decision, observation is *sequential*. In the latter, the two players don't see the other's choice until both have chosen, observation is *simultaneous*, but only for a single game. To make it so players observe only after every single game has played out, one can write `PrivatelyObserved(PrisonersDilemma())`. In the second case an agent starts getting information about the population before they have met everyone, in the last case agents can only learn about others after every interaction has taken place. Bugs rooted in observation dynamics have the potential to be very difficult to detect.

1. Repeated Games: The `Repeated` class is used when we want to play the same game or scenario a number of times and record the state of our agents after each play. `Repeated` uses the sofar unused third element of the tuple returned by `play`. This element contains a list of dictionaries each containing some information copied from the agents or return values of the `play` method in each run. Specifically itrecords the number of the round, copies the agents wholesale, each of the actions taken in each observation for that round, the payoffs of everyone at the end of that particular round, and a copy of every agent's beliefs and likelihood statistics. This is trivial to modify and makes it easy to collect information for plotting. Failing to judiciously use `copy` and `deepcopy` has led to sadness in the past.

1. Repeated Prisoner's Tournament: Defining our bread-and-butter game using everything we've made so far is as easy as `Repeated(10, PrivatelyObserved(PrisonersDilemma()))`. In this game everyone plays against everyone else and only once every game is finished does anyone get to observe, then it happens again, a total of 10 times.