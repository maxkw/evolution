import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product, permutations
import agents as ag
from utils import excluding_keys, softmax
from experiment_utils import experiment, plotter
from experiments import matchup_matrix_per_round
import params
from agents import WeAgent
from steady_state import evo_analysis, simulation
from steady_state import simulation, matchups_and_populations
from multiprocessing import Pool
from tqdm import tqdm
from copy import deepcopy


def complete_sim_live(player_types, start_pop, s=1, mu=0.000001, seed=0, **kwargs):
    pop_size = sum(start_pop)
    type_count = len(player_types)
    I = np.identity(type_count)
    pop = np.array(start_pop)
    assert type_count == len(pop)

    matchups, populations = matchups_and_populations(player_types, pop_size, "complete")
    matchup_pop_dicts = [
        dict(player_types=list(zip(*pop_pair)), **kwargs)
        for pop_pair in product(matchups, populations)
    ]

    payoffs = Parallel(n_jobs=params.n_jobs)(
        delayed(simulation)(**pop_dict)
        for pop_dict in tqdm(matchup_pop_dicts, disable=params.disable_tqdm)
    )

    type_to_index = dict(list(map(reversed, enumerate(sorted(player_types)))))
    original_order = np.array([type_to_index[t] for t in player_types])

    def sim(pop):
        f = simulation(list(zip(player_types, pop)), **kwargs)[-1]

        non_players = np.array(pop) == 0

        player_payoffs = f[non_players == False]
        f[non_players == False] = softmax(player_payoffs, s)
        f[non_players] = 0

        return f

    np.random.seed(seed)
    while True:
        yield pop

        fitnesses = sim(pop)
        actions = [(b, d) for b, d in permutations(range(type_count), 2) if pop[d] != 0]
        probs = []
        for b, d in actions:
            death_odds = pop[d] / pop_size
            birth_odds = (1 - mu) * fitnesses[b] + mu * (1 / type_count)
            prob = death_odds * birth_odds
            probs.append(prob)

        # The probability of not changing the population at all.
        actions.append((0, 0))
        probs.append(1 - np.sum(probs))

        probs = np.array(probs)
        action_index = np.random.choice(len(probs), 1, p=probs)[0]
        (b, d) = actions[action_index]
        pop = list(map(int, pop + I[b] - I[d]))


def complete_agent_simulation(
    generations, player_types, start_pop, s, seed=0, trials=100, **kwargs
):
    populations = complete_sim_live(
        player_types, start_pop, s, seed=seed, trials=trials, **kwargs
    )
    record = []

    # Populations is an infinite iterator so need to combine it with a
    # finite iterator which sets the number of generations to look at.
    for n, pop in zip(range(generations), populations):
        for t, p in zip(player_types, pop):
            record.append(
                {"generation": n, "type": t.short_name("agent_types"), "population": p}
            )

    return pd.DataFrame(record)


@plotter(
    complete_agent_simulation, plot_exclusive_args=["data", "graph_kwargs", "stacked"]
)
def complete_sim_plot(generations, player_types, data=[], graph_kwargs={}, **kwargs):
    data["population"] = data["population"].astype(int)
    data = data[["generation", "population", "type"]].pivot(
        columns="type", index="generation", values="population"
    )
    type_order = dict(
        list(
            map(
                reversed, enumerate([t.short_name("agent_types") for t in player_types])
            )
        )
    )
    data.reindex(sorted(data.columns, key=lambda t: type_order[t]), axis=1)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    data.plot(
        ax=ax,
        legend=False,
        ylim=[0, sum(kwargs["start_pop"])],
        xlim=[0, generations],
        lw=0.5,
        **graph_kwargs,
    )

    make_legend()
    plt.xlabel("Generation")

    plt.ylabel("Count")
    sns.despine()

    plt.tight_layout()

def edit_beta(player_types, beta):
    # Change the beta of each player that has a beta
    for i, t in enumerate(player_types):
        # Check for Agent Types since we only want to change the Beta on agents that have a ToM
        if hasattr(t, "genome") and "agent_types" in t.genome:
            # Update the player's beta
            player_types[i].genome["beta"] = beta
            
            # Update the beta on their ToM models
            for j in range(len(t.genome["agent_types"])):
                if hasattr(t.genome["agent_types"][j], "genome"):
                    player_types[i].genome["agent_types"][j].genome["beta"] = beta
                    
    return player_types

def ssd_v_params(param_dict, player_types, return_rounds=False, **kwargs):
    """`param_dict`: <dict> with <string> keys that name the parameter and
    values that are lists of the parameters to range over.

    """

    records = []

    # Copy the player types because they can be modified (e.g., beta) which can corrupt other experiments
    player_types = deepcopy(player_types)

    # If we aren't returning all the rounds we can't return a per_round average
    if return_rounds == False:
        kwargs["per_round"] = False

    product_params = list(product(*list(param_dict.values())))
    for pvs in tqdm(product_params, disable=params.disable_tqdm):
        ps = dict(list(zip(param_dict, pvs)))

        if "beta" in ps:
            edit_beta(player_types, ps["beta"])

        expected_pop_per_round = evo_analysis(
            player_types=player_types, **dict(kwargs, **ps)
        )
        
        # Compute the self-payoffs in the direct game
        if kwargs["game"] == "direct":
            # Delete the unnecessary parameters so that we get a cache hit on `matchup_matrix_per_round`
            combined_kwargs = dict(kwargs, **ps)
            for key in ["analysis_type", "pop_size", "s"]:
                del combined_kwargs[key]

            payoffs = matchup_matrix_per_round(
                player_types=player_types, **combined_kwargs
            )

        # Only return all of the rounds if return_rounds is True
        if return_rounds:
            start = 1
            # Need to delete the rounds key from the param dict so that it doesn't overwrite it below.
            if "rounds" in ps:
                del ps["rounds"]
        else:
            start = len(expected_pop_per_round) - 1

        for r, pop in enumerate(expected_pop_per_round, start=start):
            for t_idx, (t, p) in enumerate(zip(player_types, pop)):
                records.append(
                    dict(
                        {
                            "rounds": r,
                            "type": t.short_name("agent_types"),
                            "proportion": p,
                        },
                        **ps,
                    )
                )
                
                # Add the self payoff if we are in the direct game
                if kwargs["game"] == "direct":
                    records[-1]["selfpayoff"] = payoffs[r - start][1][t_idx][t_idx]
                    records[-1]["wepayoff"] = payoffs[r - start][1][t_idx][-1]


    return pd.DataFrame(records)


def ssd_bc(ei_stop, observe_param, delta, player_types, **kwargs):
    kwargs["per_round"] = False
    WA_index = player_types.index(WeAgent)

    records = []
    for (b, c), ei_stop in tqdm(ei_stop.items(), disable=params.disable_tqdm):
        ei = 1
        for o in tqdm(observe_param, disable=params.disable_tqdm):
            while ei <= ei_stop:
                ps = dict(observability=o, rounds=ei, benefit=b, cost=c)
                expected_pop = evo_analysis(
                    player_types=player_types, **dict(kwargs, **ps)
                )
                # There is only one round so just pop it out.
                expected_pop = expected_pop[-1]

                # If WeAgent has both the largest share of any agents
                # Second line is break ties -- if they are all equal it must be higher than equal
                WA_expected_pop = expected_pop[WA_index]
                if (WA_expected_pop == max(expected_pop)) and (
                    WA_expected_pop > 1 / len(player_types)
                ):

                    records.append(
                        dict(
                            observability=o,
                            rounds=ei,
                            benefit=b,
                            cost=c,
                            proportion=WA_expected_pop,
                        )
                    )

                    break
                ei += delta
            else:
                raise Exception("Hit the stop. Raise trials")

    return pd.DataFrame(records)


@plotter(ssd_bc)
def bc_plot(
    ei_stop, observe_param, delta, player_types, data=[], graph_kwargs={}, **kwargs
):
    data["benefit"] = data["benefit"].astype("category")

    fig, ax = plt.subplots(figsize=(3.5, 3))

    sns.pointplot(
        x="observability", y="rounds", hue="benefit", data=data, ax=ax
    )
    plt.xlabel("Probability of observation")
    plt.ylabel("Expected Interactions")
    sns.despine()
    plt.tight_layout()


@plotter(ssd_v_params)
def params_heat(
    param_dict, player_types, line=False, data=[], graph_kwargs={}, **kwargs
):
    original_data = data.copy()

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop("data")
        x = args[0]
        y = args[1]
        proportion = args[2]

        try:
            d = data.pivot(index=y, columns=x, values=proportion)
        except Exception as e:
            print("`param_dict` likely has duplicate values")
            raise e

        ax = sns.heatmap(d, **kwargs)
        ax.invert_yaxis()

        nonlocal original_data, line
        
        if line:
            # Find the first index in d where the value is greater than 0.5
            original_data.groupby([y, x]).max()

            # The type with the highest proportion had this much proportion
            max_proportion = (
                original_data.groupby([x, y], as_index=False)
                .max()
                .pivot(index=y, columns=x, values=proportion)
            )

            # The first y where the proportion is equal to the max proportion
            first = d.eq(max_proportion).idxmax().reset_index()[0]

            # Get the index integer of d that matches first
            first = d.index.get_indexer(d.eq(max_proportion).idxmax().reset_index()[0])
            
            # For the ones that never hit the max prop, make it a super higher number to get it off the plot
            first[~d.eq(max_proportion).max()] = len(d.index)*2
                        
            # Double the first entry to make the first step
            first = [first[0]] + list(first)
            plt.step(
                range(len(first)),
                first,
                color="red",
                linewidth=1,
            )

    assert len(param_dict) == 2

    graph_params = dict(
        cbar=True,
        square=True,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        xticklabels=2,
        yticklabels=2,
    )

    if graph_kwargs["onlyRA"]:
        fig, ax = plt.subplots(figsize=(3.5, 3))
        draw_heatmap(
            graph_kwargs["xy"][0],
            graph_kwargs["xy"][1],
            "proportion",
            data=data[data["type"].str.contains("WeAgent")],
            cmap=plt.cm.Blues,
            **graph_params,
        )

    else:
        g = sns.FacetGrid(data=data, col="type")
        g.map_dataframe(
            draw_heatmap,
            graph_kwargs["xy"][0],
            graph_kwargs["xy"][1],
            "proportion",
            cmap=plt.cm.gray_r,
            **graph_params,
        )

    plt.xlabel(graph_kwargs["xlabel"])
    plt.ylabel(graph_kwargs["ylabel"])
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.tight_layout()


def make_legend():
    from params import AGENT_NAME

    legend = plt.legend(frameon=True, framealpha=1)
    for i, texts in enumerate(legend.get_texts()):
        if "WeAgent" in texts.get_text():
            texts.set_text(AGENT_NAME)
        elif "SelfishAgent" in texts.get_text():
            texts.set_text("Selfish")
        elif "AltruisticAgent" in texts.get_text():
            texts.set_text("Altruistic")

    return legend


@plotter(
    ssd_v_params,
    plot_exclusive_args=[
        "experiment",
        "data",
        "legend",
        "graph_kwargs",
        "graph_funcs",
    ],
)
def selfpayoff_param_plot(
    param_dict,
    player_types,
    data=[],
    legend=True,
    graph_funcs=None,
    graph_kwargs={},
    **kwargs
):
    assert 'game' == 'direct'
    
    

@plotter(
    ssd_v_params,
    plot_exclusive_args=[
        "experiment",
        "data",
        "stacked",
        "legend",
        "graph_kwargs",
        "graph_funcs",
        "var"
    ],
)
def limit_param_plot(
    param_dict,
    player_types,
    data=[],
    stacked=False,
    var='proportion',
    legend=False,
    graph_funcs=None,
    graph_kwargs={},
    **kwargs
):
    # This function only plots a single parameter
    assert len(param_dict) == 1
    param = list(param_dict.keys())[0]
    param_values = param_dict[param]
    
    fig, ax = plt.subplots(figsize=(3.5, 3))

    if kwargs.get("return_rounds", False) and param != "rounds":
        data = data[data["rounds"] == kwargs["rounds"]]
        
    data = data[[param, var, "type"]].pivot(
        columns="type", index=param, values=var
    )

    type_order = dict(
        list(
            map(
                reversed, enumerate([t.short_name("agent_types") for t in player_types])
            )
        )
    )

    if param == "beta":
        # Need to merge the WeAgents if the parameter is Beta
        WeAgent_columns = list(filter(lambda x: "WeAgent" in x, data.columns))
        merge_into = WeAgent_columns[0]

        for c in WeAgent_columns:
            if c == merge_into:
                continue

            data[merge_into] = data[merge_into].combine_first(data[c])
            data = data.drop(columns=[c])

        # Need to rename the column to the type in the type_order list
        rename = list(filter(lambda x: "WeAgent" in x, type_order))[0]
        data.rename(columns={merge_into: rename}, inplace=True)

    data.reindex(sorted(data.columns, key=lambda t: type_order[t]), axis=1)

    if stacked:
        data.plot.bar(
            stacked=True,
            ax=ax,
            width=0.99,
            ylim=[0, 1],
            legend=False,
            linewidth=0,
            **graph_kwargs,
        )
        if legend:
            make_legend()

        if param == "rounds" and kwargs['game'] == "direct":
            ax.set_xticks(range(4, param_values[0], 5))
            ax.set_xticklabels(
                range(1, param_values[0] + 1)[4::5], rotation="horizontal"
            )

        elif param == "tremble":
            ax.set_xticks(range(0, len(param_values), 2))
            ax.set_xticklabels(param_values[::2], rotation="horizontal")
        else:
            ax.set_xticks(range(0, len(param_values), 2))
            ax.set_xticklabels(param_values[::2], rotation="horizontal")

    else:
        data.plot(ax=ax, legend=False, **graph_kwargs)
        if legend:
            make_legend()
            
    if "xlabel" in graph_kwargs:
        plt.xlabel(graph_kwargs["xlabel"])

    elif param in ["rounds", "expected_interactions"]:
        plt.xlabel("Mean Pairwise Interactions")
        # plt.xlabel("Expected Interactions\n" r"$1/(1-\gamma)$")

        # if param == 'expected_interactions':
        # plt.xticks(range(1,11))

    elif param == "observability":
        # plt.xlabel("Probability of observation\n" r"$\omega$")
        plt.xlabel(r"Prob. of observation ($\omega$)")

    elif param == "tremble":
        plt.xlabel(r"Prob. of action error ($\epsilon$)")

    else:
        plt.xlabel(param)

    if var == "proportion":
        plt.yticks([0, 0.5, 1])
        plt.ylabel("Relative abundance")
    else:
        pass

    if graph_funcs is not None:
        graph_funcs(ax)

    sns.despine()
    plt.tight_layout()


if __name__ == "__main__":
    pass
