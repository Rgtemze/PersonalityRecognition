import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 24
self_link = [(i, i) for i in range(num_node)]
#face_index = [(6, 9), (54, 55), (22, 44), (30, 41), (21, 28), (34, 52), (40, 41), (18, 19), (17, 25), (34, 35), (40, 60), (30, 51), (18, 30), (38, 40), (53, 64), (44, 45), (25, 26), (0, 17), (48, 59), (49, 58), (27, 28), (28, 31), (5, 11), (4, 5), (19, 39), (29, 30), (26, 54), (52, 63), (21, 40), (19, 24), (27, 39), (57, 59), (22, 39), (21, 37), (14, 64), (28, 32), (27, 40), (22, 42), (55, 56), (29, 37), (35, 46), (8, 57), (17, 18), (33, 34), (30, 58), (36, 41), (18, 25), (17, 39), (23, 28), (6, 60), (31, 51), (61, 63), (35, 52), (18, 36), (7, 57), (9, 30), (39, 42), (5, 64), (15, 16), (64, 66), (45, 54), (13, 26), (47, 64), (1, 15), (24, 46), (4, 12), (19, 44), (0, 36), (3, 17), (26, 44), (62, 65), (58, 59), (21, 27), (20, 21), (28, 42), (30, 46), (34, 53), (4, 48), (6, 55), (54, 64), (38, 41), (44, 46), (39, 44), (0, 18), (48, 60), (11, 26), (26, 30), (66, 67), (50, 51), (27, 31), (50, 62), (4, 11), (45, 64), (51, 63), (15, 45), (5, 10), (19, 38), (52, 56), (35, 64), (5, 7), (4, 17), (57, 58), (32, 49), (3, 60), (27, 43), (28, 33), (7, 17), (22, 43), (55, 59), (59, 60), (8, 58), (31, 33), (63, 65), (60, 63), (9, 59), (17, 38), (42, 43), (31, 50), (36, 37), (61, 62), (18, 37), (6, 64), (36, 48), (14, 26), (10, 26), (42, 64), (45, 53), (49, 50), (14, 60), (0, 15), (50, 57), (1, 14), (15, 42), (4, 13), (51, 57), (25, 46), (46, 53), (26, 45), (2, 39), (59, 64), (20, 22), (23, 43), (21, 23), (30, 34), (30, 49), (17, 44), (7, 49), (6, 59), (10, 54), (43, 44), (42, 60), (17, 49), (44, 47), (24, 29), (45, 46), (3, 4), (51, 62), (15, 44), (19, 41), (52, 57), (5, 6), (32, 50), (20, 29), (11, 64), (27, 42), (58, 62), (28, 34), (33, 51), (29, 35), (3, 48), (61, 67), (31, 32), (30, 56), (63, 64), (31, 61), (36, 38), (7, 59), (36, 49), (18, 49), (55, 65), (3, 13), (25, 45), (53, 56), (1, 39), (57, 66), (60, 64), (19, 20), (2, 36), (27, 35), (7, 9), (59, 67), (33, 52), (20, 23), (34, 51), (23, 42), (31, 41), (21, 22), (29, 41), (35, 42), (30, 35), (35, 63), (49, 67), (7, 48), (10, 55), (43, 47), (17, 48), (31, 64), (12, 54), (9, 26), (24, 30), (60, 61), (1, 17), (56, 65), (5, 8), (2, 48), (26, 53), (6, 7), (19, 29), (52, 53), (12, 64), (32, 51), (6, 10), (47, 53), (28, 35), (2, 42), (33, 50), (29, 34), (28, 46), (61, 66), (23, 44), (16, 25), (22, 28), (5, 60), (29, 47), (9, 57), (30, 57), (16, 44), (17, 36), (31, 60), (36, 39), (37, 41), (17, 41), (7, 58), (37, 38), (3, 64), (14, 15), (55, 64), (2, 17), (0, 1), (3, 12), (25, 44), (0, 39), (58, 66), (57, 65), (20, 37), (59, 66), (7, 8), (28, 37), (22, 23), (31, 40), (29, 40), (30, 32), (55, 63), (30, 55), (35, 62), (9, 55), (8, 9), (42, 47), (35, 51), (18, 41), (43, 46), (15, 26), (39, 41), (13, 54), (65, 66), (25, 30), (12, 13), (1, 16), (63, 67), (50, 61), (56, 66), (25, 53), (24, 43), (56, 62), (20, 44), (2, 60), (1, 44), (6, 11), (5, 17), (62, 66), (53, 55), (29, 33), (28, 47), (16, 26), (23, 47), (22, 29), (29, 46), (31, 34), (9, 56), (63, 66), (16, 45), (6, 48), (37, 40), (4, 64), (23, 24), (31, 63), (38, 39), (24, 25), (39, 60), (0, 2), (14, 54), (1, 3), (28, 29), (58, 67), (19, 37), (60, 66), (21, 42), (20, 38), (46, 47), (30, 31), (1, 42), (6, 17), (21, 39), (32, 33), (30, 33), (4, 60), (22, 27), (30, 52), (52, 65), (42, 44), (8, 10), (41, 60), (31, 49), (10, 53), (18, 38), (41, 49), (39, 40), (44, 64), (0, 25), (49, 61), (13, 15), (50, 58), (14, 44), (2, 3), (24, 44), (2, 14), (51, 52), (56, 63), (46, 54), (26, 46), (57, 62), (20, 24), (53, 54), (27, 47), (62, 67), (58, 61), (22, 47), (21, 29), (29, 32), (28, 40), (7, 30), (23, 46), (31, 62), (30, 50), (36, 60), (10, 11), (18, 44), (53, 65), (55, 66), (0, 16), (49, 59), (3, 14), (1, 2), (14, 42), (5, 12), (29, 31), (52, 62), (1, 36), (0, 44), (56, 57), (60, 67), (20, 39), (19, 25), (53, 63), (33, 35), (7, 10), (21, 38), (28, 39), (31, 42), (55, 57), (13, 64), (35, 47), (8, 56), (50, 67), (31, 39), (5, 48), (36, 40), (30, 53), (9, 53), (22, 24), (23, 29), (31, 48), (18, 24), (11, 54), (35, 53), (9, 10), (8, 30), (41, 48), (18, 39), (62, 63), (11, 12), (44, 54), (48, 49), (64, 65), (49, 60), (13, 14), (12, 26), (14, 45), (31, 35), (46, 64), (2, 15), (10, 64), (19, 30)]
body_index = [[ 1,  4], [ 1,  0], [ 2,  5], [ 2,  0], [ 3,  6], [ 3,  0], [ 4,  7], [ 5,  8], [ 6,  9], [ 7, 10], [ 8, 11], [ 9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]

RIGHT_PART_INDICES = [1, 4, 7, 10, 13, 16, 18, 20, 22]

symmetric_body_edges = [ [index, index + 1] for index in RIGHT_PART_INDICES]
#body_index = body_index + symmetric_body_edges
hand_index = [[0, 1], [1, 2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,13],[13,17],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
left_hand_index = [[i + 24, j + 24] for [i, j] in hand_index]
right_hand_index = [[i + 24 + 21, j + 24 + 21] for [i, j] in hand_index]

# Original
# original_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
inward = body_index
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
