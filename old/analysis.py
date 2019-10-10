from agents import AllC, AllD, WeAgent
from itertools import product
import numpy as np
from games import BinaryDictator
from params import default_genome
from copy import deepcopy
from utils import normalized

def recursive_sim(model1, model2, game, rounds):
    if rounds == 0:
        return 0
    actions = game.actions
    likelihoods1 = list(zip(actions,model1.decide_likelihood(game,"AB",game.tremble)))
    likelihoods2 = list(zip(actions,model2.decide_likelihood(game,"BA",game.tremble)))

    likelihoods = []
    payoffs = []
    for n,((A1,L1), (A2,L2)) in enumerate(product(likelihoods1, likelihoods2)):
        observations = [(game,"AB","AB", A1),
                       (game,"BA","AB", A2)]
        m1 = deepcopy(model1)
        m1.observe(observations)
        m2 = deepcopy(model2)
        m2.observe(observations)

        print("\t"*(10-rounds),n,A1,A2)
        
        payoffs.append(game(A1)+game(A2)[np.array((1,0))]+recursive_sim(m1,m2,game,rounds-1))
        likelihoods.append(L1*L2)

    likelihoods = normalized(likelihoods)

    return np.sum(p*l for p,l in zip(payoffs,likelihoods))

def matchup_sim(type1,type2,game = BinaryDictator(cost = 1,benefit= 3,tremble = 0),rounds = 10,**kwargs):
    [m1,m2] = [t(default_genome(agent_type = type1,**kwargs),a_id) for a_id,t in zip("AB",[type1,type2])]
    return recursive_sim(m1,m2,game,rounds)/rounds

print(matchup_sim(WeAgent,WeAgent,agent_types = ('self',AllC,AllD)))
