import pdb
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from experiment_utils import multi_call, experiment, plotter, MultiArg
import numpy as np
from params import default_genome
from world import World
import agents as ag
from agents import (
    WeAgent,
)
from agents import (
    leading_8_dict,
    shorthand_to_standing,
)
from games import (
    RepeatedPrisonersTournament,
    BinaryDictator,
    Repeated,
    PrivatelyObserved,
    Symmetric,
)
from collections import defaultdict
from itertools import product, permutations, combinations_with_replacement
import matplotlib.pyplot as plt
from numpy import array
from copy import copy, deepcopy
from utils import softmax_utility, _issubclass, normalized, memoize
import operator
import games


@multi_call(unordered=["player_types", "agent_types"], verbose=3)
@experiment(verbose=3)
def matchup(player_types, game, **kwargs):
    # np.random.seed(kwargs["trial"])
    
    believed_types = kwargs.get("believed_types", None)

    try:
        types, pop = list(zip(*player_types))
    except:
        # print(Warning("player_types is not a zipped list"))
        pop = tuple(1 for t in player_types)
        types = player_types

    params = dict(player_types=types, **kwargs)

    try:
        g = games.__dict__[game](**kwargs)
    except KeyError as e:
        try:
            g = game(**kwargs)
        except err:
            for n, k in enumerate(sorted(games.__dict__.keys())):
                print(n, k)
            if game in list(games.__dict__.keys()):
                raise Exception(
                    "Game must be a valid game or must be specified in the scope of 'games.py'"
                )
            else:
                raise err

    params["games"] = g

    player_types = []
    for t, p in zip(types, pop):
        player_types.extend([t] * p)

    genomes = [default_genome(agent_type=t, **params) for t in player_types]
    world = World(params, genomes)
    fitness, history = world.run()

    beliefs = []
    for agent in world.agents:
        try:
            beliefs.append(agent.belief)
        except:
            beliefs.append(None)

    record = []
    ids = [a.world_id for a in world.agents]

    # This is the special case for doing "direct" i.e., IPD style
    # analysis very efficiently. It should only apply when we aren't
    # doing per_round analysis but are doing IPD. One example of this
    # is the tremble condition for the IPD.
    if type(g) == type(RepeatedPrisonersTournament()) and kwargs['per_round'] == False:
        for t, f in zip(player_types, fitness):
            record.append({
                "player_types": tuple(player_types),
                "type": repr(t),
                "fitness": f,
                # "round": kwargs['rounds']
            })
        return record
    
    # This is the case of the `game_engine` problems. We want the "end
    # values" of fitness over types and don't care about rounds. The
    # key things we need are the total number of interactions per type
    # and the total fitness per type. 
    elif kwargs['per_round'] == False:
        # We have the summed fitnesses per type but need the summed
        # interactions and decisions per type which requires iterating
        # through the event log.
        interactions = np.zeros(len(player_types))
        decisions = np.zeros(len(player_types))
        for event in history:
            for actors in event["actors"]:
                interactions[np.array(actors)] += 1
                decisions[actors[0]] += 1
            
        for t, a_id, f in zip(player_types, ids, fitness):
            record.append({
                    "player_types": tuple(player_types),
                    "type": repr(t),
                    "id": a_id,
                    "interactions": interactions[a_id],
                    "decisions": decisions[a_id],
                    "fitness": f,
            })
            
        return record
    
    
    for event in history:
        try:
            assert len(player_types) == len(ids) and len(player_types) == len(
                event["payoff"]
            )
        except AssertionError:
            raise Warning(
                "There are %s players but the number of payoffs is %s"
                % (len(player_types), len(event["payoff"]))
            )

        r = event["round"]

        interactions = np.zeros(len(player_types))
        decisions = np.zeros(len(player_types))
        for actors in event["actors"]:
            interactions[np.array(actors)] += 1
            decisions[actors[0]] += 1
            
        for t, a_id, p, b, l, n_l in zip(
            player_types,
            ids,
            event["payoff"],
            event["beliefs"],
            event["likelihoods"],
            event["new_likelihoods"],
        ):
            if believed_types:
                if not b:
                    continue
                for o_id in ids:
                    if o_id == a_id:
                        continue

                    for believed_type in believed_types:
                        attr_to_val = {
                            "belief": b[o_id][
                                genomes[a_id]["agent_types"].index(believed_type)
                            ]
                        }

                        for attr, val in attr_to_val.items():
                            record.append(
                                {
                                    "player_types": tuple(player_types),
                                    "type": repr(t),
                                    "id": a_id,
                                    "round": r,
                                    "attribute": attr,
                                    "value": val,
                                    "believed_type": repr(believed_type),
                                    "actual_type": repr(player_types[o_id]),
                                    "fitness": p,
                                }
                            )

            else:
                # assert 0
                record.append(
                    {
                        "player_types": tuple(player_types),
                        "type": repr(t),
                        "id": a_id,
                        "round": r,
                        # "interactions": interactions[a_id],
                        # "decisions": decisions[a_id],
                        "fitness": p,
                    }
                )

    return record


def beliefs(believer, opponent_types, believed_types, **kwargs):
    """ Use this for the private belief games"""
    dfs = []
    b_name = repr(believer)  # .short_name('agent_types')
    for opponent in tqdm(opponent_types):
        data = matchup(
            player_types=(believer, opponent),
            # actual_type = repr(opponent),
            believed_types=believed_types,
            per_round=True,
            unpack_beliefs=True,
            **kwargs
        )
        
        if believer == opponent:
            dfs.append(data[data["id"] == 0])
        else:
            # idx = pd.IndexSlice
            # dfs.append(data.loc[idx[:, :, b_name, :], :])
            dfs.append(data[data["type"] == b_name])

    return pd.concat(dfs, ignore_index=True)


def population_beliefs(believer, opponent_types, believed_types, population, **kwargs):
    """ Use this for the public belief games"""
    player_types = list(zip(opponent_types, population))
    data = matchup(
        player_types=player_types,
        believed_types=believed_types,
        per_round=True,
        unpack_beliefs=True,
        **kwargs
    )

    return data


@plotter(beliefs, plot_exclusive_args=["data", "colors", "traces"])
def plot_beliefs(
    believer, opponent_types, believed_types, traces=50, colors=None, data=[], **kwargs
):
    if kwargs["observability"] != 0 and kwargs["observability"] != 1:
        raise Warning("Observability must be 0 or 1 for the axes to make sense")

    WA_prior = believer.genome["prior"]
    non_WA_prior = (1 - WA_prior) / (len(believed_types) - 1)
    prior_arr = [WA_prior if t is believer else non_WA_prior for t in believed_types]

    type_names = list(map(repr, believed_types))
    prior = dict(list(zip(type_names, prior_arr)))
    max_trials = kwargs['trials'] - 1
    
    prior_data = [
        {
            "trial": t,
            "round": 0,
            "value": prior[believed],
            "believed_type": believed,
            "actual_type": actual,
        }
        for t, believed, actual in product(list(range(max_trials)), type_names, type_names)
    ]

    color = {t: c for t, c in zip(type_names, colors)}

    prior_dat = pd.DataFrame.from_records(prior_data)
    data = pd.concat([prior_dat, data], sort=True)

    # Rescale the x-axis so that its number of interaction with
    # each type in the case where the population is specified.
    if "population" in kwargs:
        data["round"] = data["round"] / len(believed_types)
        
        # for plotting, drop the data points that aren't whole numbers. 
        whole_rounds = data['round'].round().unique()
        data = data[data["round"].isin(whole_rounds)]
        

    def name(t_n):
        if "We" in t_n:
            return "Reciprocal"
        if "Reciprocal" in t_n:
            return "Reciprocal"
        if "Selfish" in t_n:
            return "Selfish"
        if "Altruistic" in t_n:
            return "Altruistic"

        # Default
        return str(t_n)

    fig, axes = plt.subplots(figsize=(8*len(type_names)/3, 3))
    axes = {t: plt.subplot(1, len(type_names), type_names.index(t) + 1) for t in type_names}
    for (believed, actual), d in data.groupby(["believed_type", "actual_type"]):
        ax = axes[actual]

        dm = d.groupby("round").mean().reset_index()
        dm.plot(
            x="round",
            y="value",
            ax=ax,
            ylim=(-0.05, 1.05),
            yticks=[0,.25,.5,.75,1],
            xlim=(0, d["round"].max()),
            title="vs %s" % name(actual),
            label=name(believed),
            kind="scatter",
            legend=(actual == type_names[-1]),
            linewidth=2,
            color=color[believed],
        )


    # NOTE: this for-loop should not be combined with the above
    # because it will screw up the legend.
    for (believed, actual), d in data.groupby(["believed_type", "actual_type"]):
        ax = axes[actual]
        
        for trial, t in d.groupby(["trial"]):
            if trial >= traces:
                break

            t = t.groupby("round").mean().reset_index()
            t.plot(
                x="round",
                y="value",
                ax=ax,
                legend=False,
                label='_nolegend_',
                color=color[believed],
                linestyle="-",
                marker='.',
                alpha=0.1,
            )

        
        if actual == type_names[0]:
            ax.set_ylabel("Average Belief")
        else:
            ax.set_ylabel("")

        if "population" in kwargs:
            ax.set_xlabel("Avg. Observations")
        else:
            ax.set_xlabel("Pairwise Interactions")

    sns.despine()
    plt.tight_layout()


def matchup_matrix_per_round(player_types, max_rounds, cog_cost=0, sem=False, **kwargs):
    player_combos = MultiArg(combinations_with_replacement(player_types, 2))
    all_data = matchup(player_combos, rounds=max_rounds, **kwargs)

    per_round = kwargs['per_round']
    if per_round:
        groupby_keys = ["player_types", "type", "round"]
    else:
        groupby_keys = ["player_types", "type"]

    means = all_data.groupby(groupby_keys)["fitness"].mean()

    # Do the calculations for computing standard errors of the mean.
    # When going by round need to do it by summing up variances when
    # not per_round can just compute SEM directly.
    var_mean = all_data.groupby(groupby_keys)["fitness"].var()
    sem_mean = all_data.groupby(groupby_keys)["fitness"].sem()
    payoffs_var = np.zeros((len(player_types),) * 2)
    payoffs_var_list = []

    player_combos = means.index.levels[0]
    index = dict(list(map(reversed, enumerate(player_types))))

    payoffs_list = []
    payoffs = np.zeros((len(player_types),) * 2)
    
    if per_round:
        rounds = list(range(1, max_rounds + 1))
    else:
        rounds = [max_rounds]

    for r in rounds:
        for combination in player_combos:
            for players in set(permutations(combination)):
                player, opponent = players
                p, o = tuple(index[t] for t in players)

                if "WeAgent" in str(player):
                    c = cog_cost
                else:
                    c = 0

                if per_round:
                    payoffs[p, o] += means[(combination, player, r)] - c
                    payoffs_var[p, o] += var_mean[(combination, player, r)]
                        
                else:
                    try:
                        payoffs[p, o] = means[(combination, player)] - c * max_rounds
                        payoffs_var[p, o] = sem_mean[(combination, player)]
                    except:
                        import pdb; pdb.set_trace()
    
        payoffs_list.append(copy(payoffs))
        
        # TODO: These two formula should be equivalent but they are not!
        if per_round:
            # Need to change the variances into SEs
            payoffs_var_list.append(np.sqrt((payoffs_var / r ** 2) / kwargs["trials"]))
        else:
            payoffs_var_list.append(payoffs_var)

    for r, p in zip(rounds, payoffs_list):
        p /= r

    if sem:
        return list(zip(rounds, payoffs_list)), list(zip(rounds, payoffs_var_list)) 

    return list(zip(rounds, payoffs_list))
        
@plotter(matchup_matrix_per_round, plot_exclusive_args=["data"])
def payoff_heatmap(player_types, max_rounds, cog_cost=0, sem=True, data=[], **kwargs):
    # Get the last round
    data, data_sem = data[0][-1][1], data[1][-1][1]
    fig, ax = plt.subplots()
    im = ax.imshow(
        data,
        vmin=min(-kwargs["cost"], data[:].min()),
        vmax=max(kwargs["benefit"], data.max()),
        cmap="viridis_r",
    )
    cbar = ax.figure.colorbar(im, ax=ax)

    names = []
    for t in player_types:
        if "We" in str(t):
            names.append("Reciprocal")
        else:
            names.append(str(t))

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(player_types)))
    ax.set_yticks(np.arange(len(player_types)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.xaxis.tick_top()

    # Loop over data dimensions and create text annotations.
    for i in range(len(player_types)):
        for j in range(len(player_types)):
            text = ax.text(
                j,
                i,
                "%0.3f\n%0.3f" % (data[i, j], data_sem[i, j]),
                ha="center",
                va="center",
                color="w",
                fontsize=10,
            )

    fig.tight_layout()


# def test_standing(**kwargs):
#     from games import Symmetric

#     standing_types = tuple(
#         s_type
#         for name, s_type in sorted(leading_8_dict().iteritems(), key=lambda x: x[0])
#     )
#     # standing_types = (shorthand_to_standing('ggggbgbbnnnn'),)
#     standing_types = (standing_types[0],)
#     W = WeAgent(agent_types=("self",) + standing_types, RA_prior=0.49, beta=5)
#     tremble = 0
#     decision = BinaryDictator(cost=1, benefit=3, tremble=tremble)
#     game = Repeated(10, Symmetric(PrivatelyObserved(decision)))
#     # game = Repeated(10,PrivatelyObserved(Symmetric(decision)))
#     p_types = (W,) + standing_types
#     # matchup_plot(player_types = (W,)+standing_types, games = game,tremble = .05)
#     # for t in range(1,10):
#     plot_beliefs(W, p_types, p_types, file_name="WA_beliefs_Leading8", games=game)
#     for t in range(1, 10):
#         plot_beliefs(
#             W,
#             p_types,
#             p_types,
#             file_name="WA_beliefs_Leading8 - trial " + str(t),
#             games=game,
#             trials=[t],
#         )
#     #    plot_beliefs(W,(W,),(W,),file_name = 'self_belief'+str(t), games = game, trials = [t])

#     L1 = leading_8_dict()["L1"]
#     L1 = shorthand_to_standing("ggggbgbbnnnn")
#     a = L1(default_genome(agent_type=L1), "A")
#     print a.image["B"]
#     print a.decide_likelihood(decision, "AB")
#     a.observe([(decision, "BC", "ABC", "keep")])
#     print a.image["B"]
#     print a.decide_likelihood(decision, "AB")


if __name__ == "__main__":
    pass
