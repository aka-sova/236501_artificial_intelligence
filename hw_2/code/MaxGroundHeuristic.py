import math

class MaxGroundHeuristic:
    def __init__(self, board=None, player_loc=None, opp_loc=None):
        self.board = board
        self.player_loc = player_loc
        self.opp_loc = opp_loc
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.search_limit = 10 # limit the radius for calculations
        self.use_limit = 1

    def evaluate(self):
            player_ground = 0
            opp_ground = 0

            opp_paths_len = self.shortest_paths(self.opp_loc)
            player_paths_len = self.shortest_paths(self.player_loc)

            for row in range(len(opp_paths_len)):
                for col in range(len(opp_paths_len[0])):
                    if opp_paths_len[row][col] == float("inf"):
                        opp_paths_len[row][col] = -1
            for row in range(len(player_paths_len)):
                for col in range(len(player_paths_len[0])):
                    if player_paths_len[row][col] == float("inf"):
                        player_paths_len[row][col] = -1




            for row in range(len(player_paths_len)):
                for col in range(len(player_paths_len[0])):
                    if self.board[row][col] != 0:
                        continue #we don't care about the ones that are not 0
                    player_path_len = player_paths_len[row][col]
                    opp_path_len = opp_paths_len[row][col]

                    if player_path_len > 0 and opp_path_len > 0:
                        if player_path_len > opp_path_len:

                            # longer distance from player! it's opponent ground.
                            opp_ground += 1

                        elif player_path_len < opp_path_len:

                            # player ground
                            player_ground += 1

                    elif player_path_len > 0 and opp_path_len < 0:
                        player_ground += 1
                    elif player_path_len < 0 and opp_path_len > 0:
                        opp_ground += 1
            return player_ground - opp_ground


    def min_dist(self,dist,queue):
            minimum = float("Inf")
            min_index = (-1,-1)
            for i in range(len(dist)):
                for j in range(len(dist[0])):
                    if dist[i][j]  < minimum and (i,j) in queue:
                        minimum = dist[i][j]
                        min_index = (i,j)
            return min_index


    def shortest_paths(self,src_loc):
            row = len(self.board)
            col = len(self.board[0])
            dist= []
            for i in range(row):
                nrow = []
                for j in range(col):
                    nrow.append(float("Inf"))
                dist.append(nrow)
            dist[src_loc[0]][src_loc[1]] = 0
            queue = []
            for i in range(row):
                for j in range(col):
                    pos = (i,j)
                    if self.board[i][j] == 0 and self.get_euclidean_dist(src_loc, (i,j)) < self.search_limit:
                        queue.append(pos)
            queue.append(src_loc)

            while queue:
                u = self.min_dist(dist,queue)
                if u not in queue:
                    return dist
                queue.remove(u)
                for move in self.directions:
                    i = u[0] + move[0]
                    j = u[1] + move[1]
                    adj = (i,j)
                    if adj in queue:
                        if dist[u[0]][u[1]] + 1 < dist[i][j]:
                            dist[i][j] = dist[u[0]][u[1]] + 1
            return dist

    def get_euclidean_dist(self, src, trg):

        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(src, trg)]))
