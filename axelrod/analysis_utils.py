from axelrod.moran import moran
import matplotlib.pyplot as plt

def joined_stats_plot(mp):
    _, (top,bot) = plt.subplots(2,1)
    top.set_ylim(0,1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    top.plot(mp.blind_history, label="Information")
    top.plot(mp.coop_history, label="Cooperation")
    top.plot([len(pop)/len(mp.populations[0]) for pop in mp.populations], label="Num strategies")
    top.set_title("Moran Process Statistics, Information cost = " + str(int(mp.modifier)))
    top.set_xlabel("Iteration")
    top.set_ylabel("Ratio")
    top.grid()
    top.legend(loc='upper left')
    
    # Bottom
    last_index = 0
    top_player_count = 10
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
