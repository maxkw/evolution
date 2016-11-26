import pandas as pd
from utils import multi_call
import numpy as np
from indirect_reciprocity import ReciprocalAgent
from utils import is_sequency

print is_sequency(np.linspace(.1,.9,5))

### all expermints need to have the same 
@multi_call
def fitness_v_selfish_thing(RA_K = [1], proportions = [round(n,5) for n in np.linspace(.1,.9,5)], N_agents = 50, visibility = "private", observability = .5, trial = 10, RA_prior = .80, p_tremble = 0, agent_type = ReciprocalAgent, rounds = 10, **kwargs):
    args = kwargs["expected args"]
    #print args
    return True

fitness_v_selfish_thing()
