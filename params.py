import seaborn as sns

sns.set_style('ticks')
sns.set_context('paper', font_scale=1)

n_jobs = 8
disable_tqdm = False
memoized = True

AGENT_NAME = 'Bayesian Reciprocator'

def default_genome(agent_type = False, agent_types = None, prior = .5, **extra_args):
    try:
        agent_types = agent_type.genome['agent_types']
    except:
        pass
    
    if agent_types:
        agent_types = tuple(t if t != 'self' else agent_type for t in agent_types)

    genome = {
        'type': agent_type,
        'prior': prior,
        "agent_types": agent_types,
    }

    for key in extra_args:
        if key in genome and key != 'prior':
            genome[key] = extra_args[key]

    return genome
