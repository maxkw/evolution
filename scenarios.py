import pandas as pd
from collections import Counter, OrderedDict
import seaborn as sns
from experiment_utils import experiment, plotter, save_fig
import numpy as np
from params import default_genome
from agents import AltruisticAgent, SelfishAgent, WeAgent
from games import BinaryDictator
import matplotlib.pyplot as plt
from itertools import product


def lookup_agent(a):
    a = str(a)
    if "WeAgent" in a or "self" in a:
        return "Reciprocal"
    elif "AltruisticAgent" in a:
        return "Altruistic"
    elif "SelfishAgent" in a:
        return "Selfish"
    else:
        return a
        # raise "String not defined for agent %s"


agent_to_label = {
    AltruisticAgent: "Altruistic",
    SelfishAgent: "Selfish",
    WeAgent: "Reciprocal",
}

@experiment(memoize=False)
def scenarios(scenarios, **kwargs):
    record = []

    for name, observations in enumerate(scenarios):
        og = default_genome(agent_type=kwargs["observer"], **kwargs["observer"].genome)
        observer = og["type"](og, "O")

        for observation in observations:
            observer.observe([observation])

        for agent_type in observer._type_to_index:
            record.append(
                {
                    "player_types": None,
                    "scenario": name,
                    "belief": observer.belief_that("A", agent_type),
                    "type": lookup_agent(agent_type),
                }
            )

    return record


@plotter(scenarios, plot_exclusive_args=["data", "color", "graph_kwargs"])
def scene_plot(
    titles=None,
    xlabel=None,
    data=[],
    color=sns.color_palette(["C5", "C0", "C1"]),
    graph_kwargs={},
):
    sns.set_context("talk", font_scale=0.8)

    order = ["Reciprocal", "Altruistic", "Selfish"]
    f_grid = sns.catplot(
        data=data,
        y="type",
        x="belief",
        col="scenario",
        kind="bar",
        orient="h",
        order=order,
        palette=color,
        aspect=1.2,
        height=1.8,
        sharex=False,
        sharey=False,
        **graph_kwargs
    )

    if titles is not None and "Prior" in titles[0]:
        f_grid.set_xlabels(" ")
    else:
        # f_grid.set_xlabels("Belief")
        f_grid.set_xlabels(xlabel)

    (
        f_grid.set_titles("")
        .set_ylabels("")
        .set(
            xlim=(0, 1),
            xticks=np.linspace(0, 1, 5),
            xticklabels=["0", "", "0.5", "", "1"],
            yticklabels=["B", "A", "S"],
        )
    )

    if titles is not None:
        for i, (t, ax) in enumerate(zip(titles, f_grid.axes[0])):
            if i > 0:
                ax.set_yticklabels([])
            # ax.set_title(t, fontsize=14, loc='right')
            ax.set_title(t, loc="right")

    plt.tight_layout()


def make_observations_from_scenario(scenario, **kwargs):
    """
    Scenario is a dict with keys `actions` and `title`
    """
    letter_to_action = {"C": "give", "D": "keep"}
    observations = list()

    game = BinaryDictator(**kwargs)
    for players, action in zip(*scenario):
        observations.append(
            {
                "game": game,
                "action": letter_to_action[action],
                "participant_ids": players,
                "observer_ids": "ABO",
            }
        )

    return observations

def forgive_plot(p, game_params, scenarios, label="", **kwargs):
    game = BinaryDictator(**game_params)
    og = default_genome(agent_type=kwargs["observer"], **kwargs["observer"].genome)
    record = []

    obs_dict = {"game": game, "observer_ids": "ABO"}
    for trial in scenarios:
        
        observer = og["type"](og, "O")
        
        for act in trial:
            if act == "C":
                observer.observe(
                    [
                        {
                            "action": "give",
                            "participant_ids": "AB",
                            **obs_dict
                        },
                        {
                            "action": "give",
                            "participant_ids": "BA",
                            **obs_dict
                        },
                    ]
                )
            elif act == "D":
                observer.observe(
                    [
                        {
                            "action": "keep",
                            "participant_ids": "AB",
                            **obs_dict
                        },
                        {
                            "action": "give",
                            "participant_ids": "BA",
                            **obs_dict
                        },                    
                    ]
                )

        record.append(
            {
                "cooperations": trial,
                "belief": observer.belief_that("A", og["type"]),
                "decision": observer.decide_likelihood(game, "BA", 0)[1],
                "type": "after",
                "player_types": None,
            }
        )

    # Plot Data
    data = pd.DataFrame(record)
    fplot = sns.catplot(
        data=data,
        y="cooperations",
        x=p,
        orient="h",
        # bw=.1,
        palette={"k"},
        kind="bar",
        hue="type",
        legend=False,
        aspect=1,
    )

    # fplot.axes[0][0].axhline(.33, 0, 2, color='black')
    fplot.set(
        # ylim=(0, 1.05),
        xlabel=label,
        ylabel="",
        # xticklabels=["0", "0.5", "1"],
        xticks=[0, 0.5, 1],
    )

    # if p == "decision":
    #     fplot.set(yticklabels=[])

    plt.tight_layout()
    save_fig(**kwargs)

def decision_plot(game_params, betas = None, **kwargs):
    game = BinaryDictator(**game_params)

    
    if betas is None:
        betas = [kwargs["observer"].genome['beta']]
        
    record = []
    ticks = 100
    # Could also add a beta flag here
    
    for b in betas: 
        kwargs["observer"].genome['beta'] = b
        og = default_genome(agent_type=kwargs["observer"], **kwargs["observer"].genome)
        agent = og['type'](og, 'A')
        tindx = agent._type_to_index[og["type"]]
        
        for i in np.linspace(0,1,ticks):
            agent.models['A'].belief['B'][tindx] = i
            record.append({
                "belief": i,
                "decision": agent.decide_likelihood(game, "AB", 0)[1],
                'beta': b,
            })
        
    data = pd.DataFrame(record)
    data = data.pivot(index='belief', columns='beta', values='decision')
    fig, ax = plt.subplots(figsize=(3.5, 3))
    data.plot(ax=ax)
    sns.despine()
    save_fig(**kwargs)