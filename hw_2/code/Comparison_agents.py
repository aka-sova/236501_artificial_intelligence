
from MapsGenerator import ai_board
import numpy as np

from MinimaxPlayer import MinimaxPlayer
from AlphaBetaPlayer import AlphaBetaPlayer
from OrderedAlphaBetaPlayer import OrderedAlphaBetaPlayer
import matplotlib.pyplot as plt

# Configurations
names = ['Minimax', 'AlphaBeta', 'OrderedAlphaBeta']
colors = ['r', 'g', 'b']
player_classes = [MinimaxPlayer, AlphaBetaPlayer, OrderedAlphaBetaPlayer]

names_pruning = ['AlphaBeta', 'OrderedAlphaBeta']
# colors = ['g', 'b']
# player_classes = [AlphaBetaPlayer, OrderedAlphaBetaPlayer]

time_arr_1 = np.linspace(0.1, 3, 10)
time_arr_2 = np.linspace(3, 30, 5)

time_arr = np.concatenate((time_arr_1, time_arr_2), axis = None)

time_arr = time_arr_1

fig = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()
fig6 = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)
ax5 = fig5.add_subplot(111)
ax6 = fig6.add_subplot(111)

ax1.set_xlabel('Time')
ax1.set_ylabel('Depth')
ax1.set_title('Depth reach in limited time')
ax1.grid()

ax2.set_xlabel('Depth')
ax2.set_ylabel('Heuristics used')
ax2.set_title('Heuristics used for each agent')
ax2.grid()


ax3.set_xlabel('Depth')
ax3.set_ylabel('Branches pruned')
ax3.set_title('Pruning of branches vs depth')
ax3.grid()


ax4.set_xlabel('Depth')
ax4.set_ylabel('Leaves created')
ax4.set_title('Leaves created vs depth')
ax4.grid()

ax5.set_xlabel('Time')
ax5.set_ylabel('Best value returned')
ax5.set_title('Best value vs time')
ax5.grid()

ax6.set_xlabel('Depth')
ax6.set_ylabel('Best value returned')
ax6.set_title('Best value vs depth')
ax6.grid()

for name, color, player_class in zip(names, colors, player_classes):
    times = []
    depths = []
    branches_pruned = []
    heuristics_used = []
    leaves_developed = []
    max_values = []


    for t in time_arr:
        print(f"time: {round(t,2)}")
        player = player_class()
        player.set_game_params(ai_board.copy())
        d, max_val = player.make_move(t)

        times.append(t)
        depths.append(d)
        max_values.append(max_val)


        heuristics_used.append(player.heuristics_used)
        leaves_developed.append(player.leaves_developed)

        if name in ['AlphaBeta', 'OrderedAlphaBeta']:
            branches_pruned.append(player.branches_pruned)

    if name in ['AlphaBeta', 'OrderedAlphaBeta']:
        ax3.scatter(depths, branches_pruned, c=color, label=name)

        
    ax1.plot(times, depths, c=color, label=name)
    ax2.scatter(depths, heuristics_used, c=color, label=name)
    ax4.scatter(depths, leaves_developed, c=color, label=name)

    ax5.plot(times, max_values, c=color, label=name)
    ax6.scatter(depths, max_values, c=color, label=name)
    

    

ax1.legend(names, loc='upper left')
ax2.legend(names, loc='upper left')
ax3.legend(names_pruning, loc='upper left')
ax4.legend(names, loc='upper left')
ax5.legend(names, loc='upper left')
ax6.legend(names, loc='upper left')

plt.show()