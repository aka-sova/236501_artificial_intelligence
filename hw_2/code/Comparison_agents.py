
from MapsGenerator import ai_board
import numpy as np

from MinimaxPlayer import MinimaxPlayer
from AlphaBetaPlayer import AlphaBetaPlayer
from OrderedAlphaBetaPlayer import OrderedAlphaBetaPlayer
import matplotlib.pyplot as plt

agents_to_test = [AlphaBetaPlayer]

for agent in agents_to_test:

    times = []
    depths = []

    for t in np.linspace(0.1, 3, 50):
        player = agent()
        player.set_game_params(ai_board.copy())
        d = player.make_move(t)
        times.append(t)
        depths.append(d)

    plt.scatter(times, depths)
    plt.show()