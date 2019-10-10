
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class World(object):
    def __init__(self, params, genomes):
        self.agents = []
        self.id_to_agent = {}

        self.add_agents(genomes)
        self.tremble = params['tremble']
        self.game = params['games']
        self.params = params

    def add_agents(self, genomes):
        self.agents = list(self.agents)
        for world_id, genome in enumerate(genomes):
            agent = genome['type'](genome, world_id)
            self.agents.append(agent)
            self.id_to_agent[world_id] = agent
        self.agents = np.array(self.agents)

    def run(self):
        agents = np.array(self.agents)
        payoff, observations, record = self.game.play(agents, agents, tremble=self.params['tremble'])
        
        return payoff, record
