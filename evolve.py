import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product, permutations
from scipy.ndimage.filters import gaussian_filter1d
import agents as ag
from utils import excluding_keys, softmax, memoize
from experiment_utils import experiment, plotter
import params
from agents import WeAgent
from steady_state import evo_analysis, simulation
from steady_state import simulation, matchups_and_populations
from multiprocessing import Pool
from tqdm import tqdm


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

    @memoize
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
        **graph_kwargs
    )

    make_legend()
    plt.xlabel("Generation")

    plt.ylabel("Count")
    sns.despine()

    plt.tight_layout()


def ssd_v_param(param, player_types, **kwargs):
    """
    This should be optimized to reflect the fact that
    in terms of complexity
    rounds=s>=pop_size>=anything_else

    for 'rounds' you do a single analysis and then plot each round
    for 's' you should only make the RMCP once and then do the analysis for different s
    for 'pop_size',
       in direct reciprocity you only need to make the payoff matrix once
       in indirect you need to make an rmcp for each value of pop_size
    for anything else, the whole thing needs to be rerun


    """
    records = []

    if param == "rounds" and "param_vals" not in kwargs:
        kwargs["per_round"] = True
        expected_pop_per_round = evo_analysis(player_types=player_types, **kwargs)
        for r, pop in enumerate(expected_pop_per_round, start=1):
            for t, p in zip(player_types, pop):
                records.append(
                    {
                        "rounds": r,
                        "type": t.short_name("agent_types"),
                        "proportion": p,
                    }
                )

        return pd.DataFrame(records)

    if "param_vals" in kwargs:
        vals = kwargs["param_vals"]
        del kwargs["param_vals"]

        return_rounds = kwargs.get("return_rounds", False)
        kwargs["per_round"] = kwargs.get("return_rounds", False)

        for x in tqdm(vals, disable=params.disable_tqdm):
            if param == "beta":
                # Change the beta of each player that has a beta
                for i, t in enumerate(player_types):
                    if hasattr(t, "genome") and "beta" in t.genome:
                        player_types[i].genome["beta"] = x

            expected_pop = evo_analysis(
                player_types=player_types, **dict(kwargs, **{param: x})
            )

            if return_rounds:
                for r, pop in enumerate(expected_pop, start=1):
                    for t, p in zip(player_types, pop):
                        records.append(
                            {
                                "rounds": r,
                                param: x,
                                "type": t.short_name("agent_types"),
                                "proportion": p,
                            }
                        )
            else:
                assert len(expected_pop) == 1

                expected_pop = expected_pop[-1]
                for t, p in zip(player_types, expected_pop):
                    records.append(
                        {
                            param: x,
                            "type": t.short_name("agent_types"),
                            "proportion": p,
                        }
                    )

        return pd.DataFrame(records)

    else:
        raise Exception("`param_vals` %s is not defined. Pass this variable" % param)


def ssd_v_params(param_dict, player_types, return_rounds=False, **kwargs):
    """`param_dict`: <dict> with <string> keys that name the parameter and
    values that are lists of the parameters to range over.

    """

    records = []

    if return_rounds == False:
        kwargs["per_round"] = False

    product_params = list(product(*list(param_dict.values())))
    for pvs in tqdm(product_params, disable=params.disable_tqdm):
        ps = dict(list(zip(param_dict, pvs)))
        expected_pop_per_round = evo_analysis(
            player_types=player_types, **dict(kwargs, **ps)
        )

        # Only return all of the rounds if return_rounds is True
        if return_rounds:
            start = 1
        else:
            start = len(expected_pop_per_round) - 1

        for r, pop in enumerate(expected_pop_per_round[start:], start=start):
            for t, p in zip(player_types, pop):
                records.append(
                    dict(
                        {
                            "rounds": r,
                            "type": t.short_name("agent_types"),
                            "proportion": p,
                        },
                        **ps
                    )
                )

    return pd.DataFrame(records)


def ssd_bc(ei_stop, observe_param, delta, player_types, **kwargs):
    kwargs["per_round"] = False
    WA_index = player_types.index(WeAgent)

    records = []
    for (b, c), ei_stop in tqdm(ei_stop.items(), disable=params.disable_tqdm):
        ei = 1
        for o in tqdm(observe_param, disable=params.disable_tqdm):
            while ei <= ei_stop:
                ps = dict(observability=o, expected_interactions=ei, benefit=b, cost=c)
                expected_pop = evo_analysis(
                    player_types=player_types, **dict(kwargs, **ps)
                )
                # There is only one round so just pop it out.
                expected_pop = expected_pop[-1]

                # If WeAgent has both the largest share of any agents
                # Second line is break ties -- if they are all equal it must be higher than equal
                WA_expected_pop = expected_pop[WA_index]
                if WA_expected_pop == max(expected_pop) and WA_expected_pop > 1 / len(
                    player_types
                ):

                    records.append(
                        dict(
                            observability=o,
                            expected_interactions=ei,
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
        x="observability", y="expected_interactions", hue="benefit", data=data, ax=ax
    )
    plt.xlabel("Probability of observation")
    plt.ylabel("Expected Interactions")
    sns.despine()
    plt.tight_layout()


def ssd_v_xy(x_param, y_param, x_vals, y_vals, player_types, **kwargs):
    return ssd_v_params(
        params={x_param: x_vals, y_param: y_vals}, player_types=player_types, **kwargs
    )


@plotter(ssd_v_params)
def params_heat(param_dict, player_types, data=[], graph_kwargs={}, **kwargs):
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop("data")
        try:
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
        except Exception as e:
            print("`param_dict` likely has duplicate values")
            raise e

        ax = sns.heatmap(d, **kwargs)
        ax.invert_yaxis()

    assert len(param_dict) == 2

    if graph_kwargs["onlyRA"]:
        fig, ax = plt.subplots(figsize=(3.5, 3))
        draw_heatmap(
            graph_kwargs["xy"][0],
            graph_kwargs["xy"][1],
            "proportion",
            data=data[data["type"].str.contains("WeAgent")],
            cbar=True,
            square=True,
            vmin=0,
            vmax=1,
            # vmax=data['frequency'].max(),
            cmap=plt.cm.Blues,
            linewidths=0.5,
        )

    else:
        g = sns.FacetGrid(data=data, col="type")
        g.map_dataframe(
            draw_heatmap,
            graph_kwargs["xy"][0],
            graph_kwargs["xy"][1],
            "proportion",
            cbar=False,
            square=True,
            vmin=0,
            vmax=1,
            # vmax=data['frequency'].max(),
            cmap=plt.cm.gray_r,
            linewidths=0.5,
        )

    plt.xlabel(graph_kwargs["xlabel"])
    plt.ylabel(graph_kwargs["ylabel"])
    plt.tight_layout()


@plotter(ssd_v_param)
def beta_heat(param_vals, player_types, data=[], graph_kwargs={}, **kwargs):
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop("data")
        try:
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
        except Exception as e:
            print("`param_dict` likely has duplicate values")
            raise e

        ax = sns.heatmap(d, **kwargs)
        ax.invert_yaxis()

    if graph_kwargs["onlyRA"]:
        fig, ax = plt.subplots(figsize=(3.5, 3))
        draw_heatmap(
            graph_kwargs["xy"][0],
            graph_kwargs["xy"][1],
            "proportion",
            data=data[data["type"].str.contains("WeAgent")],
            cbar=True,
            square=True,
            vmin=0,
            vmax=1,
            # vmax=data['frequency'].max(),
            cmap=plt.cm.Blues,
            linewidths=0.5,
            xticklabels=2,
            yticklabels=2,
        )

    plt.xlabel(graph_kwargs["xlabel"])
    plt.ylabel(graph_kwargs["ylabel"])
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.tight_layout()


def ssd_param_search(
    param, param_lim, player_types, target, param_tol, mean_tol, **kwargs
):

    min_val, max_val = param_lim

    def is_mode(ssd):
        return sorted(zip(ssd, player_types))[-1][1] == target

    def tolerable(ssd):
        y, z = sorted(ssd)[-2:]
        return z - y <= mean_tol

    @memoize
    def get_ssd(val):
        return evo_analysis(player_types=player_types, **dict(kwargs, **{param: val}))[
            -1
        ]

    min_p = get_ssd(min_val)
    if is_mode(min_p):
        if tolerable(min_p):
            best = min_val
        else:
            best = min_val  # "poor min"
    else:
        max_p = get_ssd(max_val)
        if not is_mode(max_p):
            best = max_val
        elif tolerable(max_p):
            best = max_val
        else:

            def finder(max_val, min_val):
                mid_val = np.round(np.mean((max_val, min_val)), 5)
                # check tolerance
                if max_val - mid_val <= param_tol:
                    return max_val
                mid_p = get_ssd(mid_val)
                if is_mode(mid_p):
                    if tolerable(mid_p):
                        return mid_val
                    else:
                        return finder(mid_val, min_val)
                else:
                    return finder(max_val, mid_val)

            best = finder(max_val, min_val)

    ret = dict(
        kwargs,
        **{
            param: best,
            "proportion": get_ssd(best),
            "player_types": player_types,
            "type": target,
        }
    )

    return ret


def make_legend():
    legend = plt.legend(frameon=True, framealpha=1)
    for i, texts in enumerate(legend.get_texts()):
        if "WeAgent" in texts.get_text():
            texts.set_text("Reciprocal")
        elif "SelfishAgent" in texts.get_text():
            texts.set_text("Selfish")
        elif "AltruisticAgent" in texts.get_text():
            texts.set_text("Altruistic")

    return legend


@plotter(
    ssd_v_param,
    plot_exclusive_args=[
        "experiment",
        "data",
        "stacked",
        "legend",
        "graph_kwargs",
        "graph_funcs",
    ],
)
def limit_param_plot(
    param,
    player_types,
    data=[],
    stacked=False,
    legend=True,
    graph_funcs=None,
    graph_kwargs={},
    **kwargs
):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    # TODO: Investigate this, some weird but necessary data cleaning
    data[data["proportion"] < 0] = 0
    # data[data["proportion"]] = 0
    data = data[data["type"] != 0]

    data = data[[param, "proportion", "type"]].pivot(
        columns="type", index=param, values="proportion"
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
        merge_into = list(filter(lambda x: "WeAgent" in x, type_order))[0]
        for c in WeAgent_columns:
            if c == merge_into:
                continue

            data[merge_into] = data[merge_into].combine_first(data[c])
            data = data.drop(columns=[c])

    data.reindex(sorted(data.columns, key=lambda t: type_order[t]), axis=1)

    if stacked:
        data.plot.bar(
            stacked=True,
            ax=ax,
            width=0.99,
            ylim=[0, 1],
            legend=False,
            linewidth=0,
            **graph_kwargs
        )
        if legend:
            make_legend()

        if param == "rounds":
            ax.set_xticks(range(4, kwargs["rounds"], 5))
            ax.set_xticklabels(
                range(1, kwargs["rounds"] + 1)[4::5], rotation="horizontal"
            )

        elif param == "tremble":
            plt.xticks(rotation="horizontal")

        else:
            ax.set_xticks(range(0, len(kwargs["param_vals"]), 2))
            ax.set_xticklabels(kwargs["param_vals"][::2], rotation="horizontal")

        if param in ["rounds", "expected_interactions"]:
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

    else:

        data.plot(ax=ax, ylim=[0, 1.05], **graph_kwargs)

        if param in ["pop_size"]:
            plt.axes().set_xscale("log", basex=2)
        elif param == "s":
            plt.axes().set_xscale("log")

        # elif param in ["beta"]:
        #     plt.axes().set_xscale('log',basex=10)
        # elif if param in ['rounds']:
        #     pass

        plt.xlabel(param)
        plt.legend()

    plt.yticks([0, 0.5, 1])
    plt.ylabel("Equilibrium Frequency")

    if graph_funcs is not None:
        graph_funcs(ax)

    sns.despine()
    plt.tight_layout()


@plotter(ssd_v_xy)
def param_v_rounds_heat(
    x_param, y_param, x_vals, y_vals, player_types, data=[], **kwargs
):
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop("data")
        d = data.pivot(index=y_param, columns=x_param, values="proportion")
        ax = sns.heatmap(d, **kwargs)
        ax.invert_yaxis()

    g = sns.FacetGrid(data=data, col="type")
    g.map_dataframe(
        draw_heatmap,
        cbar=False,  # square=True,
        vmin=0,
        vmax=1,
        # vmax=data['frequency'].max(),
        cmap=plt.cm.gray_r,
    )

    # linewidths=.5)


if __name__ == "__main__":
    pass
