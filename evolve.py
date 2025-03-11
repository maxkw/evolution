import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from experiment_utils import plotter
import params
from agents import WeAgent
from steady_state import evo_analysis
from tqdm import tqdm
from copy import deepcopy
from iteround import saferound

plot_exclusive_args=[
        "experiment",
        "data",
        "line",
        "stacked",
        "legend",
        "graph_kwargs",
        "graph_funcs",
        "var",
        "add_payoff",
        "xy"
    ]

def edit_genome(player_types, key, value):
    # Change the beta of each player that has a beta
    for i, t in enumerate(player_types):
        # Only change the genome if they have `key` in it
        if hasattr(t, "genome") and key in t.genome:
            # Update the player's genome
            player_types[i].genome[key] = value

        # Recursively update the genome of the agent_types for agents that have it
        if "agent_types" in t.genome:
            for j in range(len(t.genome["agent_types"])):
                if hasattr(t.genome["agent_types"][j], "genome"):
                    player_types[i].genome["agent_types"][j].genome[key] = value

    return player_types    

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

def multipop_ssd_v_params(param_dict, player_types, return_rounds=False, **kwargs):
    dfs = []
    for types in player_types:
        dfs.append(ssd_v_params(param_dict, types, return_rounds, **kwargs))
        
    return dfs        

def ssd_v_params(param_dict, player_types, return_rounds=False, **kwargs):
    """`param_dict`: <dict> with <string> keys that name the parameter and
    values that are lists of the parameters to range over.

    """

    records = []

    # Copy the player types because they can be modified (e.g., beta) which can corrupt other experiments
    player_types = deepcopy(player_types)

    # If we aren't returning all the rounds we can't return a per_round average
    if not return_rounds:
        kwargs["per_round"] = False

    product_params = list(product(*list(param_dict.values())))
    for pvs in tqdm(product_params, disable=params.disable_tqdm):
        ps = dict(list(zip(param_dict, pvs)))

        if "beta" in ps:
            player_types = edit_beta(player_types, ps["beta"])
            
        if "w_aia" in ps or "w_dia" in ps:
            player_types = edit_genome(player_types, "w_aia", ps["w_aia"])
            player_types = edit_genome(player_types, "w_dia", ps["w_dia"])
            
        if "prior" in ps:
            player_types = edit_genome(player_types, "prior", ps["prior"])

        expected_pop_per_round, payoffs = evo_analysis(
            player_types=player_types, **dict(kwargs, **ps)
        )

        # # Compute the self-payoffs in the direct game
        # if "direct" in kwargs["game"] and kwargs["analysis_type"] == "limit":
        #     # Delete the unnecessary parameters so that we get a cache hit on `matchup_matrix_per_round`
        #     combined_kwargs = dict(kwargs, **ps)
        #     for key in ["analysis_type", "pop_size", "s"]:
        #         del combined_kwargs[key]

        #     payoffs = matchup_matrix_per_round(
        #         player_types=player_types, **combined_kwargs
        #     )
        # elif kwargs["analysis_type"] == "complete":
        #     pass
        

        # Only return all of the rounds if return_rounds is True
        if return_rounds:
            start = 1
            # Need to delete the rounds key from the param dict so that it doesn't overwrite it below.
            if "rounds" in ps:
                del ps["rounds"]
        else:
            start = len(expected_pop_per_round) - 1

        for r, pop in enumerate(expected_pop_per_round, start=start):
            # Compute the total payoff for the rounded steady state distribution
            if "direct" in kwargs["game"]:
                total_payoff = 0
                rounded_ssd = saferound(pop*kwargs['pop_size'], 0)
                for i in range(len(pop)):
                    if rounded_ssd[i] == 0: 
                        continue 
                    
                    for j in range(len(pop)):
                        if rounded_ssd[j] == 0: 
                            continue
                        
                        total_payoff += payoffs[r - start][1][i][j] * rounded_ssd[i] * rounded_ssd[j] 
                        
                total_payoff = total_payoff / kwargs['pop_size']**2
                ps["total_payoff"] = total_payoff 
                
                # Normalize total_payoff by the benefit and cost, only do this for direct games
                if 'benefit' in ps and ps["benefit"]>kwargs['cost']:
                    ps["total_payoff"] = ps["total_payoff"] /  (ps['benefit'] - kwargs['cost'])
                elif 'benefit' in kwargs and kwargs['benefit']>kwargs['cost']:
                    ps["total_payoff"] = ps["total_payoff"] / (kwargs['benefit'] - kwargs['cost'])   
                                 
            else:
                # These will look like they are diveded by 2 since the game engine is not symmetric so each "round" is only one player interacting with another
                ps["total_payoff"] = payoffs
                    
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
                if "direct" in kwargs["game"]:
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
                expected_pop, _ = evo_analysis(
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


@plotter(ssd_bc, plot_exclusive_args=plot_exclusive_args)
def bc_plot(
    ei_stop, observe_param, delta, player_types, data=[], graph_kwargs={}, **kwargs
):
    data["benefit"] = data["benefit"].astype("category")

    fig, ax = plt.subplots(figsize=(3.5, 3))

    sns.pointplot(x="observability", y="rounds", hue="benefit", data=data, ax=ax)
    ax.set_xlabel(r"Prob. observation ($\omega$)")
    ax.set_ylabel("Game length")
    ax.set_yticks(range(1,6))
    sns.despine()
    plt.tight_layout()


@plotter(ssd_v_params, plot_exclusive_args=plot_exclusive_args)
def params_dom_heat(
        param_dict, player_types, data=[], graph_kwargs={}, **kwargs
):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    x = graph_kwargs["xy"][0]
    y = graph_kwargs["xy"][1]
    
    # drop rows where the round is even
    data = data[data['rounds'] % 2 == 1]
    xs = data[x].unique()
    ys = data[y].unique()
    player_strings = sorted(list(data['type'].unique()), key=str)
    vmap = {i: player_strings[i] for i in range(len(player_strings))}
    n = len(vmap)

    heat = np.zeros((len(ys), len(xs)))
    for (i, j) in product(range(len(ys)), range(len(xs))):
        max_proportion = data[
            (data[y] == ys[i]) & (data[x] == xs[j])]["proportion"].max()
     
        dominant_type = data[
            (data[y] == ys[i]) & (data[x] == xs[j]) & (data["proportion"] == max_proportion)]["type"].iloc[0]
        
        # Clean up this numerical instability where Extort2 and AllD should be the same with 1 round. 
        # if ys[i] == 1 and 'Extort2' in dominant_type:
        #     dominant_type = 'AllD'
            
        # If the sum of AllD and Extort2 is greater than the max_proportion then the dominant type is AllD. 
        p_alld = data[(data[y] == ys[i]) & (data[x] == xs[j]) & (data['type']=='AllD')]['proportion']
        p_extort = data[(data[y] == ys[i]) & (data[x] == xs[j]) & (data['type']=='Extort2')]['proportion']
        if p_alld.sum() + p_extort.sum() > max_proportion:
            dominant_type = 'AllD'


        
        
        
        
    
        # # Check if AllD and Extort2 are both in the type field
        # non_partner_proportion = 0
        # if 'AllD' and 'Extort2' in data['type'].unique():
        #     # Sum the proportion of AllD and Extort2
        #     non_partner_proportion = data[
        #         (data[y] == ys[i]) & (data[x] == xs[j]) & (data["type"].str.contains('AllD') | data["type"].str.contains('Extort2'))]["proportion"].sum()        

        # # if the non-partner strategies are dominant then there won't be cooperation    
        # import pdb; pdb.set_trace()
        # if max_proportion < non_partner_proportion:
        #     dominant_type = 'AllD'
            
        dominant_index = player_strings.index(dominant_type)

        # print(ys[i], xs[j], max_proportion, dominant_type, dominant_index)
        heat[i, j] = dominant_index
    
    # Fill downward from the first AllD (handles approximate ties when there is no cooperation)
    
    # first_indx = heat.shape[0]-np.argmax(np.flipud(heat)==1, axis=0)
    # for i in range(heat.shape[1]):
    #     heat[:first_indx[i], i] = 1    
    
    sns.heatmap(heat, cmap=graph_kwargs['color'], ax=ax, linewidths=0.1,vmin=0, vmax=len(player_strings)-1, square=False,
                cbar=False, 
                )
    ax.invert_yaxis()
    colors = ax.collections[0].get_facecolors()
    ax.collections[0].set_edgecolors(colors)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    sns.despine(fig=fig, top=True, right=True)
    
    rotation = 0
    if x == "tremble":
        param_values = data[x].unique()
        ax.set_xticks(np.arange(3.5, len(param_values), 4))
        ax.set_xticklabels(param_values[3::4], rotation=rotation)   
    elif x == "benefit":
        param_values = data[x].unique()
        ax.set_xticks(np.arange(4.5, len(param_values), 5))
        ax.set_xticklabels(param_values[4::5], rotation=rotation)            
    else:
        param_values = data[x].unique()
        ax.set_xticks(np.arange(.5, len(param_values), 5))
        ax.set_xticklabels(param_values[0::5], rotation=rotation)             
            
    if y == "rounds":
        tick_increment = 8
        skip = 2
        max_rounds = data[y].max()
        ax.set_ylim([0,max_rounds/skip+.5])
        ax.set_yticks(np.arange(.5, max_rounds/skip+.5, tick_increment/skip))
        ax.set_yticklabels(np.arange(1,max_rounds+1,tick_increment), rotation=0)    
    
    plt.xlabel(graph_kwargs["xlabel"])
    plt.ylabel(graph_kwargs["ylabel"])
    
    plt.tight_layout()        

@plotter(ssd_v_params, plot_exclusive_args=plot_exclusive_args)
def params_heat(
    param_dict, player_types, line=True, data=[], graph_kwargs={}, **kwargs
):
    original_data = data.copy()
    direct = (kwargs['game'] == 'direct' or kwargs['game'] == 'direct_seq')
    if direct:
        # Drop rows where the round is even
        data = data[data['rounds'] % 2 == 1]
                    

    # Need to merge the FSAgents if the parameter is w_aia or w_dia
    if "w_aia" in param_dict or "w_dia" in param_dict:
        data.loc[data['type'].str.contains('FS'), 'type'] = 'FSAgent'          

    def draw_heatmap(*args, **kwargs):
        nonlocal line, direct

        data = kwargs.pop("data")
        x = args[0]
        y = args[1]
        proportion = args[2]

        try:
            d = data.pivot(index=y, columns=x, values=proportion)
        except Exception as e:
            print("`param_dict` likely has duplicate values")
            raise e

        ax = sns.heatmap(
            d,
            **dict(
                **kwargs,
            ),
        )
        ax.invert_yaxis()
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        sns.despine(ax=ax, top=True, right=True)
        
        rotation = 0 if direct else 45
        
        if x == "tremble":
            param_values = data[x].unique()
            
            # This is a hacky switch where less than 10 are the engine experiments and greater than 10 are the ipd experiments
            if not direct:
                ax.set_xticks(np.arange(.5, len(param_values), 2))
                ax.set_xticklabels(param_values[0::2], rotation=rotation)
            else:
                ax.set_xticks(np.arange(3.5, len(param_values), 4))
                ax.set_xticklabels(param_values[3::4], rotation=rotation)
        elif x == "benefit":
            param_values = data[x].unique()
            ax.set_xticks(np.arange(4.5, len(param_values), 5))
            ax.set_xticklabels(param_values[4::5], rotation=rotation)
            
            
        if y == "rounds" and direct:
            tick_increment = 8
            max_rounds = data[y].max()
            ax.set_ylim([0,max_rounds/2+.5])
            ax.set_yticks(np.arange(.5, max_rounds/2+.5, tick_increment/2))
            ax.set_yticklabels(np.arange(1,max_rounds+1,tick_increment), rotation=0)                

        # # Change the font size of the label
        # ax.figure.axes[-1].yaxis.label.set_size(8)
        # # Make the label on top of the colorbar
        # ax.figure.axes[-1].set_title('Abundance', size=6)


        nonlocal original_data

        # Draw the line. 
        if proportion == "proportion" and line:
            # Find the first index in d where the value is greater than 0.5
            # original_data.groupby([y, x]).max()

            # The type with the highest proportion had this much proportion
            max_proportion = (
                original_data.groupby([x, y], as_index=False)
                .max()
                .pivot(index=y, columns=x, values=proportion)
            )

            # The first y where the proportion is equal to the max proportion
            first = (d>.5).idxmax().reset_index()[0]
            # first = (d>max_proportion).idxmax().reset_index()[0]

            # Get the index integer of d that matches first
            first = d.index.get_indexer(first)

            # For the ones that never hit the max prop, make it a super higher number to get it off the plot
            first[~d.eq(max_proportion).max()] = len(d.index) * 2

            # Double the first entry to make the first step
            first = [first[0]] + list(first)
            plt.step(
                range(len(first)),
                first,
                color="red",
                linewidth=1,
            )

    assert len(param_dict) == 2

    if direct:
        square = False
        linewidths=0

    else:
        square = True
        linewidths=0.5
        
    graph_params = dict(
        cbar=True,
        square=square,
        linewidths=linewidths,
        xticklabels=2,
        yticklabels=2,
        cbar_kws={
            "location": "right",
            "label": "Abundance",
            "fraction": 0.1,
            "ticks": [0, 0.5, 1],
            "shrink": 0.6,
        },        
    )

    if graph_kwargs['var'] == 'proportion':
        graph_params['vmin'] = 0
        graph_params['vmax'] = 1
        graph_params['cmap'] = plt.cm.Blues
        graph_params['cbar_kws'].update({
            "label": "Abundance",
            "ticks": [0, 0.5, 1],
        })                
    elif graph_kwargs['var'] == 'total_payoff':
        graph_params['cmap'] = plt.cm.gray_r
        # graph_params['vmin'] = np.floor(data['total_payoff'].min())
        # graph_params['vmax'] = np.ceil(data['total_payoff'].max())
        graph_params['cbar_kws'].update({
            "label": "Payoff",
            "ticks": [np.ceil(data['total_payoff'].min()*10)/10, np.floor(data['total_payoff'].max()*10)/10],
        })     
        # Set the colorbar ticks to be the min and max of the total_payoff which are already normalized to 0-1
        if direct:
            graph_params['cbar_kws']["ticks"] = [0, 1]
            graph_params['vmin'] = 0
            graph_params['vmax'] = 1            
    else:
        raise Exception('Invalid var')
        
    if graph_kwargs["who"] == 'RA' or graph_kwargs['var']=='total_payoff':
        fig, ax = plt.subplots(figsize=(3.5, 3))
        
        # Need this for when doing the old_pop pick an agent type that isn't WeAgent since that agent won't be there. 
        if graph_kwargs['var'] == 'total_payoff' and direct:
            data = data[data["type"].str.match("AllD")]
        else:
            data = data[data["type"].str.contains("WeAgent")]
            
        draw_heatmap(
            graph_kwargs["xy"][0],
            graph_kwargs["xy"][1],
            graph_kwargs['var'],
            data=data,
            **graph_params,
        )
        
        plt.xlabel(graph_kwargs["xlabel"])
        plt.ylabel(graph_kwargs["ylabel"])
        plt.yticks(rotation=0)
        
    elif graph_kwargs["who"] == 'all' or graph_kwargs["who"] == 'nice':
        graph_params['cmap'] = plt.cm.Blues
        graph_params['cbar'] = False
        if graph_kwargs["who"] == 'nice':
            nice_list = dict()
            for a in player_types:
                if str(a.type) == 'Memory1PDAgent':
                    nice_list[str(a)] = a([]).nice_or_not()
                else:
                    nice_list[a.short_name('agent_types')] = str(a.type)
            
            data.loc[:, 'nice'] = data.loc[:, 'type'].map(nice_list)
            
            # Sum the proportion for rows that have the same nice type 
            data = data.groupby([graph_kwargs["xy"][0], graph_kwargs["xy"][1], 'nice']).sum().reset_index()
            
            g = sns.FacetGrid(data=data, col="nice",col_wrap=3, sharey=True, aspect=.85)
        else: 
            g = sns.FacetGrid(data=data, col="type",col_wrap=3)
        
        g.map_dataframe(
            draw_heatmap,
            graph_kwargs["xy"][0],
            graph_kwargs["xy"][1],
            graph_kwargs["var"],
            **graph_params,
        )
        g.set_xlabels(graph_kwargs["xlabel"])
        g.set_ylabels(graph_kwargs["ylabel"])
        g.set_titles(col_template="{col_name}")
    else: 
        raise Exception('Invalid who')

    plt.tight_layout()

def make_legend(ax, loc='best'):
    from params import AGENT_NAME

    legend = ax.legend(frameon=True, framealpha=1, loc=loc)
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
    plot_exclusive_args=plot_exclusive_args,
)
def payoff_plot(param_dict,
    player_types,
    var,
    data=[],
    graph_kwargs={},
    **kwargs
):
    fig, ax = plt.subplots(figsize=(3.5, 3))

    sns.lineplot(data=data, 
                 x=list(param_dict.keys())[0], 
                 y=var, 
                 ax=ax,
                 color='black', marker='o')   
    
    ax.set(**graph_kwargs)
    sns.despine()
    plt.tight_layout()

@plotter(
    ssd_v_params,
    plot_exclusive_args=plot_exclusive_args,
)
def limit_param_plot(
    param_dict,
    player_types,
    data=[],
    stacked=False,
    var="proportion",
    add_payoff = False,
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
        
    data = data[[param, var, "type"]].pivot(columns="type", index=param, values=var)
    type_order = dict(
        list(
            map(
                reversed, enumerate([t.short_name("agent_types") for t in player_types])
            )
        )
    )

    if param == "beta" or param == "prior":
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

        if param == "rounds" and "direct" in kwargs["game"]:
            ax.set_xticks(range(4, param_values[0], 10))
            ax.set_xticklabels(
                range(1, param_values[0] + 1)[4::10], rotation="horizontal"
            )

        elif param == "tremble":
            ax.set_xticks(range(3, len(param_values), 4))
            ax.set_xticklabels(param_values[3::4], rotation="horizontal")
            
        elif param == 'benefit':
            ax.set_xticks(list(range(0, len(param_values)))[4::5])
            ax.set_xticklabels(param_values[4::5], rotation="horizontal")

        else:
            ax.set_xticks(range(0, len(param_values), 2))
            ax.set_xticklabels(param_values[::2], rotation="horizontal")

    else:
        data.plot(ax=ax, legend=False, **graph_kwargs)
        
    # # Adding the payoff / cooperation % to the plot    
    # if var=='proportion' and add_payoff:
    #     payoff_data = payoff_data[[param, 'total_payoff', "type"]].pivot(columns="type", index=param, values='total_payoff')
    #     payoff_data.reindex(sorted(payoff_data.columns, key=lambda t: type_order[t]), axis=1)
        
    #     twin = ax.twinx()
    #     twin.plot(range(len(payoff_data.index)), payoff_data.iloc[:,0], color='black', linestyle='--', marker='.', label='payoff')
    #     ymax = np.ceil(payoff_data.max().iloc[0])
    #     twin.set_ylim([0, ymax])
    #     twin.set_yticks([0, ymax/2, ymax])
    #     if 'direct' in kwargs['game']:
    #         twin.set_ylabel('% Cooperate')
    #     else:
    #         twin.set_ylabel('Payoff')

    if "xlabel" in graph_kwargs:
        ax.set_xlabel(graph_kwargs["xlabel"])

    elif param in ["rounds", "expected_interactions"]:
        ax.set_xlabel("Game length")

    elif param == "observability":
        # plt.xlabel("Probability of observation\n" r"$\omega$")
        ax.set_xlabel(r"Prob. observation ($\omega$)")

    elif param == "tremble":
        ax.set_xlabel(r"Prob. action error ($\epsilon$)")

    elif param == "observation_error":
        ax.set_xlabel(r"Prob. observation error")
    else:
        ax.set_xlabel(param)

    if var == "proportion":
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Abundance")
    elif var == "total_payoff":
        ax.set_ylim([0, kwargs['benefit']-kwargs['cost']])
        ax.set_ylabel("Average Payoff")

    if graph_funcs is not None:
        graph_funcs(ax)
        
    if legend:
        legend = 'best'
        make_legend(ax, legend)
        
    sns.despine()
    plt.tight_layout()


if __name__ == "__main__":
    pass
