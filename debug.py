import original_indirect_reciprocity as old
import indirect_reciprocity as new
import numpy as np
from utils import namedArrayConstructor, pickled, unpickled
from itertools import product, combinations


from inspect import isclass
def try_to_pickle(obj):
    #if isclass(obj):
    #    pass
    try:
        pickled(obj,"debug/junk.pkl")
    except:
        print obj
        if isinstance(obj,dict):
            for key,val in obj.items():
                try_to_pickle(val)
        else:
            for key,val in obj.__dict__.items():
                try_to_pickle(val)


def unpickleable_world(world):
    world.agents = map(dict2agent,world.agents)
    world.id_to_agent = {agent.world_id:agent for agent in world.agents}
    world.stop_condition = partial(*world._stop_condition)
    return world

def run_and_pickle(n):
    old_world = old.fitness_rounds_experiment(n,overwrite = True)
    old_world.pickle('debug/old_world.pkl')
    old.fitness_rounds_plot(out_path = 'debug/old_world_fitness.pdf')

    new_world = new.fitness_rounds_experiment(n,overwrite = True)
    new_world.use_npArrays()
    new_world.pickle('debug/new_world.pkl')
    new.fitness_rounds_plot(out_path = 'debug/new_world_fitness.pdf')

#run_and_pickle(20)
def load_worlds():
    old_world = old.World.unpickle('debug/old_world.pkl')
    new_world = new.World.unpickle('debug/new_world.pkl')
    return old_world,new_world



def belief_accuracy(world,agents):
    correct = 0
    total = 0
    if type(world)== old.World:
        for believer, target in product(agents,repeat=2):
            if believer.world_id == target.world_id:
                continue
            if not isinstance(believer,old.ReciprocalAgent):
                continue
            else:
                total += 1
                believes_target_is_RA = believer.belief[target.world_id]>=.5
                target_is_RA = isinstance(target,old.ReciprocalAgent)
                if believes_target_is_RA and target_is_RA:
                    correct += 1
    else:
        for believer, target in product(agents,repeat=2):
            if believer.world_id == target.world_id:
                continue
            if not isinstance(believer,new.ReciprocalAgent):
                continue
            else:
                total += 1
                believes_target_is_RA = believer.belief[target.world_id][new.ReciprocalAgent]>=.5
                target_is_RA = isinstance(target,new.ReciprocalAgent)
                if believes_target_is_RA and target_is_RA:
                    correct += 1
    accuracy = float(correct)/total
    print accuracy
    return accuracy



    
type_convert = {
    new.ReciprocalAgent : old.ReciprocalAgent,
    new.SelfishAgent : old.SelfishAgent
    }

from copy import deepcopy
def genome_new2old(new_genome):
    old_genome = deepcopy(new_genome)
    old_genome['type'] = type_convert[old_genome['type']]
    del old_genome['prior']
    del old_genome['agent_types']
    return old_genome

def belief_new2old(prior):
    return prior[new.ReciprocalAgent]

def agent_new2old(new_agent):
    old_genome = genome_new2old(new_agent.genome)
    agent = old_genome['type'](old_genome, world_id = new_agent.world_id)
    agent.belief = {id:belief_new2old(belief) for id,belief in new_agent.belief.iteritems()}
    agent.likelihood = {id:belief_new2old(belief) for id,belief in new_agent.likelihood.iteritems()}
    if isinstance(agent,old.ReciprocalAgent):
        agent.models = {id:agent_new2old(model) for id,model in agent.models.iteritems()}
    agent_new2old_unit_test(agent,new_agent)
    return agent

def agent_new2old_unit_test(oa,na):

    def compare(od,nd):
        for id in nd.iterkeys():
            assert od[id] == nd[id][new.ReciprocalAgent]
            
    compare(oa.belief,na.belief)
    compare(oa.likelihood,na.likelihood)
    
def world_new2old(new_world):
    pass

from itertools import izip
from numpy import array

def check_beliefs():
    old_world,new_world = load_worlds()

    belief_accuracy(new_world,new_world.agents)
    belief_accuracy(old_world,old_world.agents)

def compare_agent_styles(new,old):
    """
    tests that the utility and decide_likelihood work the same in both versions
    """
    old_world = old.World.unpickle('debug/old_world.pkl')
    new_world = new.World.unpickle('debug/new_world.pkl')
    game = new_world.game
    old_game = old_world.game
    new_agents = new_world.agents
    old_agents = [agent_new2old(new_agent) for new_agent in new_agents]

    ids = [agent.world_id for agent in new_agents]



    for [new,old],id in product(izip(new_agents,old_agents),ids):
        assert new.world_id == old.world_id
        ids = [new.world_id,id]
        
        nu = array([new._utility(1,id) for id in ids])
        ou = array([old.utility(1,id) for id in ids])
        print
        print nu
        print ou
        
        utilities_agree = all(nu == ou) 
                
        nl = new.decide_likelihood(new,game,ids)
        ol = old.decide_likelihood(old,old_game,ids)
        likelihoods_agree = all(nl ==ol)

        if not (utilities_agree and likelihoods_agree):
            print "Old and New disagree on",id
            print "New Utility", nu
            print "Old Utility", ou
            print "New Likelihood",nl
            print "Old Likelihood",ol
            raise AssertionError



def compare_observations():
    """
    make two agents with the same ids, one old and one new,
    feed the same observations to see if their beliefs change in tandem

    BELIEFS DO CHANGE IN TANDEM
    """
    params = new.default_params()
    agent_types = [new.ReciprocalAgent,new.SelfishAgent]
    new_genome = {'type': new.ReciprocalAgent,
                  'RA_prior': params['RA_prior'],
                  'prior_precision': params['prior_precision'],
                  'beta': params['beta'],
                  'prior' : new.prior_generator(agent_types,params['RA_prior']),
                  'agent_types': agent_types}
    old_genome = genome_new2old(new_genome)
    new_agent = new.ReciprocalAgent(new_genome,world_id = 0)
    old_agent = old.ReciprocalAgent(old_genome,world_id = 0)

    for id in range(3):
        new_agent.belief[id] = new_agent.initialize_prior()
        old_agent.belief[id] = old_agent.initialize_prior()

    game = g = params['games']

    agents = a = range(3)
    observations =[(game, [1,2], a,'keep') for _ in range(10)]

    def update_and_print(observation):
        actor = observation[1][0]
        print "Observed:",observation
        print "prior belief that",actor,"is reciprocal"
        print "\told-style:",old_agent.belief[actor]
        print "\tnew-style:",new_agent.belief[actor][new.ReciprocalAgent]
        new_agent.observe_k([observation],2)
        old_agent.observe_k([observation],2)

        print "posterior belief that",actor,"is reciprocal"
        print "\told-style:",old_agent.belief[actor]
        print "\tnew-style:",new_agent.belief[actor][new.ReciprocalAgent]
        print
        
    for observation in observations:
        update_and_print(observation)

    for id in range(3):
        new_agent.belief[id] = new_agent.initialize_prior()
        old_agent.belief[id] = old_agent.initialize_prior()

    for actor in range(3):
        print "prior belief that",actor,"is reciprocal"
        print "\told-style:",old_agent.belief[actor]
        print "\tnew-style:",new_agent.belief[actor][new.ReciprocalAgent]
        
    new_agent.observe_k(observations,2)
    old_agent.observe_k(observations,2)

    for actor in range(3):
        print "posterior belief that",actor,"is reciprocal"
        print "\told-style:",old_agent.belief[actor]
        print "\tnew-style:",new_agent.belief[actor][new.ReciprocalAgent]

#compare_observations()
        
from numpy import random
import numpy as np

def random_test():
    random.seed(0)
    print np.where(random.multinomial(1,[1/3.0]*3))
    print random.get_state()[2]

    random.seed(0)
    print random.multinomial(1,[1/3.0]*3)
    print random.get_state()[2]

#random_test()
#raise
def parallel_worlds(rounds):
    params = new.default_params()
    agent_types = [new.ReciprocalAgent,new.SelfishAgent]
    new_genome_RA = {'type': new.ReciprocalAgent,
                     'RA_prior': params['RA_prior'],
                     'prior_precision': params['prior_precision'],
                     'beta': params['beta'],
                     'prior' : new.prior_generator(agent_types,params['RA_prior']),
                     'agent_types': agent_types}
    new_genome_SA = deepcopy(new_genome_RA)
    new_genome_SA['type'] = new.SelfishAgent

    old_genome_RA = genome_new2old(new_genome_RA)
    old_genome_SA = genome_new2old(new_genome_SA)

    
    params['stop_condition'][1] = rounds
    params['RA_K'] = 1
    new_world = new.World(params,[new_genome_RA, new_genome_RA, new_genome_SA])
    old_world = old.World(params,[old_genome_RA, old_genome_RA, old_genome_SA])

    return old_world, new_world
    
def parallel_test():
    ow,nw = parallel_worlds(10)
    owr = ow.run()
    nwr = nw.run()
    print ow.last_run_results['seeds']
    print nw.last_run_results['seeds']
    
    #print zip(ow.last_run_results['seeds'],ow.last_run_results['seeds'])
    #return
    
    
    for oo,no in zip(owr[1],nwr[1]):
        print "round",oo['round']
        print "old", 
        print "new", 
        print
        print "players:"
        print oo['pair']
        print no['pair']
        print
        print "decide_likelihood"
        print oo['likelihoods']
        print no['likelihoods']
        print
        print "actions,payoffs:"
        print oo['actions'],oo['payoff']
        print no['actions'],no['payoff']
        print
        print "observations"
        print oo['observations']
        print no['observations']
        print
        print "beliefs"
        print oo['belief']
        print no['belief']
        for x,y in combinations(oo['pair'][0],2):
            try:
                print "%s's posterior belief that %s is reciprocal" % (x,y)
                print oo['belief'][x][y]
                print no['belief'][x][y][new.ReciprocalAgent]
                print
            except KeyError:
                pass
            
        print
        print
        print

    print ow.last_run_results['fitness']
    print nw.last_run_results['fitness']
    
parallel_test()

    

    
def test_run():

    def inspect(h):
        [h] = h
        print h['payoff']
        print
    old_w, new_w = load_worlds()
    random.seed(8)
    inspect(old_w.run())
    random.seed(8)
    inspect(new_w.run())
#test_run()

        
#compare_agent_styles(new,old)
#compare_observations()

