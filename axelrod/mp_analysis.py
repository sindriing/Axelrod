import matplotlib.pyplot as plt

def get_all_stats(ms, verbose=True):
    coops, infos, modifs, top_p = [], [], [], []
    for k in ms:
        top_players = Counter()
        coop, info = [],[]
        for m in ms[k]:
            top_players += m.populations[-1]
            coop.append(m.coop_history[-1])
            info.append(m.blind_history[-1])
        modifs.append(m.modifier)
        coops.append(np.mean(coop))
        infos.append(np.mean(info))
        top_p.append(top_players.most_common(3))
        if verbose:
            print("Information cost = ", round(modifs[-1],2))
            print("Average Cooperation: ", round(coops[-1],2))
            print("Average Information: ", round((infos[-1]),2))
            print("Top Players: ")
            for p in top_players.most_common(3):
                print(p)
            print()
    return modifs, infos, coops, top_p

def joined_stats_plot(mp, top_player_count=10):
    _, (top,bot) = plt.subplots(2,1)
    top.set_ylim(0,1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    top.plot(mp.blind_history, label="Information")
    top.plot(mp.coop_history, label="Cooperation")
#     top.plot([len(pop)/len(mp.populations[0]) for pop in mp.populations], label="Num strategies")
    top.set_title("Moran Process Statistics, Information cost = " + str(round(mp.modifier,2)))
    top.set_xlabel("Iteration")
    top.set_ylabel("Ratio")
    top.grid()
    top.legend(loc='upper left')
    
    # Bottom
    last_index = 0
    for i, pop in enumerate(reversed(mp.populations)):
        if len(pop) > top_player_count:
            last_index = len(mp.populations) - 1 - i
            break

    player_names = [x[0] for x in mp.populations[last_index].most_common()]

    plot_data = []
    labels = []
    for name in player_names:
        labels.append(name)
        values = [counter[name] for counter in mp.populations]
        plot_data.append(values)
        domain = range(len(values))

    bot.stackplot(domain, plot_data, labels=labels)
    bot.set_title("Moran Process Population of by Iteration")
    bot.set_xlabel("Iteration")
    bot.set_ylabel("Number of Individuals")
    bot.legend(loc='upper left')
    return (bot,top)
