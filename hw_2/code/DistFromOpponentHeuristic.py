class DistFromOpponentHeuristic:
    def __init__(self, board=None, player_loc=None, opp_loc=None):
        self.board = board
        self.player_loc = player_loc
        self.opp_loc = opp_loc
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def evaluate(self):
        return self.shortest_path(self.player_loc, self.opp_loc)

    def shortest_path(self, source, target):
        cur_pos = source
        visited_pos = set()
        min_paths_len = {source: (None, 0)}
        while cur_pos != target:
            visited_pos.add(cur_pos)
            destinations = self.get_next_pos(cur_pos)
            len_to_cur = min_paths_len[cur_pos][1]
            for next_pos in destinations:
                cur_path_len = 1 + len_to_cur
                if next_pos not in min_paths_len:
                    min_paths_len[next_pos] = (cur_pos, cur_path_len)
                else:
                    minimum_found_len = min_paths_len[next_pos][1]
                    if minimum_found_len > cur_path_len:
                        min_paths_len[next_pos] = (cur_pos, cur_path_len)
            next_destinations = {pos: min_paths_len[pos] for pos in min_paths_len if pos not in visited_pos}
            if not next_destinations:
                return -1  # no path between source and dest
            cur_pos = min(next_destinations,
                          key=lambda k: next_destinations[k][1])  # next node is the destination with the shortest path
        return min_paths_len[target][1]

    def get_next_pos(self, pos: tuple):

        next_pos = []
        for d in self.directions:
            i = pos[0] + d[0]
            j = pos[1] + d[1]

            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and self.board[i][j] == 0:  # then move is legal
                new_loc = (i, j)
                next_pos.append(new_loc)

        return next_pos


