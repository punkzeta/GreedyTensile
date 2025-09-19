import numpy as np
import networkx as nx
import random
from math import log
from datetime import datetime
import csv

#Basic PyTorch setup
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import normalize

#Use GPU if available
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# url='model\\GreedyTensile_RL_euclidean_n50_d5_rnd19_s3.pt'
url='..\\model\\GreedyTensile_supervised_euclidean_n50_d5_rnd19_s3.pt'
output_url='GreedyTensile_test_result_euclidean.csv'

Radius = 1000
margin = 0.05
density_list = [2, 3, 4, 5]
Invalid = -1

now = datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("datetime = ", dt_string)

model_name = url.split('model')[-1][0:-3]
print(model_name)

def get_pos(G, node):
  pos = nx.get_node_attributes(G,'pos')
  return pos[node][0], pos[node][1]

def get_dist(G, node1, node2):
  x1, y1 = get_pos(G, node1)
  x2, y2 = get_pos(G, node2)
  return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def get_path_length(G, path):
  path_length = 0.0
  for i in range(len(path)-1):
    path_length += get_dist(G, path[i], path[i+1])
  return path_length

def get_ellipse_factor(G, src, dst, v):
  return (get_dist(G, src, v)+get_dist(G, v, dst))/get_dist(G, src, dst)

def dijkstra_all_pairs_shortest_path(G):


  max_dist = float('inf')
  dist = dict()
  for i in G.nodes:
    dist[i] = dict()

  path = dict()
  for i in G.nodes:
    path[i] = dict()

  for source in G.nodes:
    unvisited_set = set()

    for node in G.nodes:
      # Set the max distance from the source to all destinations
      dist[source][node] = max_dist
      path[source][node] = list()

      # Add all nodes to the unvisited set
      unvisited_set.add(node)

    # Set the distance for the start node to zero
    path[source][source].append(source)
    dist[source][source] = 0

    # Iteratively do the search of shortest path until all nodes are visited.
    while len(unvisited_set) != 0:

      min_dist = max_dist
      v = list(unvisited_set)[0]

      #select the unvisited node with min length_of_sp[node]
      for node in unvisited_set:
        if dist[source][node] < min_dist:
          min_dist = dist[source][node]
          v = node
      # print("Visit {}".format(v))
      unvisited_set.remove(v)

      # ToDo: complete the loop for updating dist[] and path[]

      # For each v's neighbor u still in unvisited_set, update dist[] and sp[] if we find a shorter path

      # Tip 1: use G[v][u]['weight'] to present the weight of a given edge
      # Tip 2: use the operation of new_path = list(path[v]) + list([u]) to update the newly found path [source, ... v, u]

      for u in G[v]:
        if u in unvisited_set:
          dist_alt = dist[source][v] + G[v][u]['weight']
          path_alt = list()
          if dist_alt != max_dist and dist_alt < dist[source][u]:
            dist[source][u] = dist_alt
            path_alt = list(path[source][v]) + list([u])
            path[source][u] = path_alt


  return path, dist

def set_graph(Size, Density, Radius, rnd_seed):

  # Set a uniform random graph based on the given size and density
  G=nx.Graph()
  gridwidth = pow(Size * Radius * Radius / Density, 0.5)
  # print('gridwidth = ', gridwidth)
  random.seed(rnd_seed)

  # Set nodes
  for i in range(Size):
    x_pos = random.uniform(0, gridwidth)
    y_pos = random.uniform(0, gridwidth)
    # print(x_pos, y_pos)
    G.add_node(i,pos=(x_pos, y_pos))

  # Set edges
  for i in range(Size):
    for j in range(i+1, Size):
      if get_dist(G, i, j) <= Radius:
        # G.add_edge(i,j)
        G.add_edge(i, j, weight=get_dist(G, i, j))

  # pos = nx.get_node_attributes(G, 'pos')
  # nx.draw_networkx_nodes(G, pos)
  # nx.draw_networkx_edges(G, pos)
  # nx.draw_networkx_labels(G, pos)
  # plt.show()

  all_pairs_sp, all_pairs_sp_length = dijkstra_all_pairs_shortest_path(G)

  return G, all_pairs_sp, all_pairs_sp_length


def shortest_path(model, G, src, dst, all_pairs_sp):
    count = 0
    dummy = 1000
    v = src
    path = [v]
    visited = {v}
    done = False

    if len(all_pairs_sp[src][dst]) == 0:
        done = True
        path = []

    while done == False:
        q_values = np.zeros(len(G.nodes))
        q_values += dummy
        # Predict the q values of the device's neighbors
        for u in G[v]:
            if u not in visited:
                x_data = torch.Tensor(get_x_data(G, v, u, src, dst)).to(dev)
                # print("train x ({}, {}) = {}".format(v, u, x_data))
                x_data = normalize(x_data, p=2.0, dim=0)
                # print("normalized train x ({}, {}) = {}".format(v, u, x_data))
                q_values[u] = model(x_data)

        count += 1
        # print("v = ", v)
        # print("q_values = ", q_values)
        # Determine the next node
        v = np.argmin(q_values)

        # Check if there is no available neighbor
        if q_values[v] == dummy or count >= len(G.nodes):
            done = True

        # print("select ", v)

        path.append(v)
        visited.add(v)

        # print("path = ", path)
        # print("visited = ", visited)

        if v == dst:
            done = True

    return path

def get_x_data(G, v, u, src, dst):
  #Fill up state features of v
  s_0 = get_dist(G, src, dst)/Radius
  s_1 = get_ellipse_factor(G, src, dst, v)
  s_2 = get_dist(G, v, dst)/Radius

  #Fill up action features of u
  a_0 = get_dist(G, src, dst)/Radius
  a_1 = get_ellipse_factor(G, src, dst, u)
  a_2 = get_dist(G, u, dst)/Radius

  return np.array([s_0, s_1, s_2, a_0, a_1, a_2])

def test_model_single(model, G, dst, all_pairs_sp, all_pairs_sp_length):
  accuracy_count = 0
  count = 0
  mre = 0.0
  accuracy_count_norm = 0
  mre_norm = 0
  miss = 0
  variance_actual = 0
  variance_predicted = 0

  l = list(G.nodes)
  l.remove(dst)

  for src in l:
    s_path = all_pairs_sp[src][dst]

    if len(s_path) != 0:
        # print('Dijkstra shortest path = ', s_path)
        actual_dist = all_pairs_sp_length[src][dst]/Radius
        # print('Dijkstra shortest path distance = ', actual_dist)

        path = shortest_path(model, G, src, dst, all_pairs_sp)
        # print('predicted shortest path = ', path)
        predict_dist = get_path_length(G, path)/Radius
        # print('predicted shortest path distance = ', predict_dist)

        path_stretch = actual_dist/(get_dist(G, src, dst)/Radius)
        # print("path_stretch = ", path_stretch)

        predicted_path_stretch = predict_dist / (get_dist(G, src, dst) / Radius)

        count += 1

        if dst not in path:
          path = []
          predict_dist = 0
          mre += 1.0
          mre_norm += 1.0
          miss += 1
        else:
          if predict_dist != 0 and predict_dist <= actual_dist:
            accuracy_count += 1
          else:
            abe = np.fabs(predict_dist - actual_dist)
            re = abe/actual_dist
            mre += re

          if predict_dist != 0 and predict_dist <= actual_dist*path_stretch*(1.0+margin):
            accuracy_count_norm += 1
          else:
            abe = np.fabs(predict_dist - actual_dist)
            re_norm = abe/(actual_dist * path_stretch)
            mre_norm += re_norm

        variance_actual += path_stretch ** 2
        variance_predicted += predicted_path_stretch ** 2
  # print('accuracy = ', accuracy_count/count)
  # print("mre:", mre/count)

  if count !=0:
    return accuracy_count/count, mre/count, accuracy_count_norm/count, mre_norm/count, miss/count, variance_actual/count, variance_predicted/count
  else:
    return Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid

def test_model(model, G, all_pairs_sp, all_pairs_sp_length):
  accuracy_count = 0.0
  mre_count = 0.0
  accuracy_count_norm = 0
  mre_count_norm = 0
  miss_count = 0
  variance_actual_count = 0
  variance_predicted_count = 0
  count = 0

  for dst in G.nodes:
    accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual, variance_predicted = test_model_single(model, G, dst, all_pairs_sp, all_pairs_sp_length)
    accuracy_count += accuracy
    mre_count += mre
    accuracy_count_norm += accuracy_norm
    mre_count_norm += mre_norm
    miss_count += miss
    count += 1

  # print('total accuracy = ', accuracy_count/count)
  # print("mean relative error:", mre_count/count)
  if count != 0:
    return accuracy_count/count, mre_count/count, accuracy_count_norm/count, mre_count_norm/count, miss_count/count, variance_actual_count/count, variance_predicted_count/count
  else:
    return Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid


def test_model_multigraph(model, size, density, radius, rnd_list):
    accuracy_count = 0.0
    mre_count = 0.0
    accuracy_count_norm = 0
    mre_count_norm = 0
    miss_count = 0
    variance_actual_count = 0
    variance_predicted_count = 0
    count = 0

    for rnd in rnd_list:
        G, all_pairs_sp, all_pairs_sp_length = set_graph(size, density, radius, rnd)
        accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual, variance_predicted = test_model(model, G, all_pairs_sp, all_pairs_sp_length)
        accuracy_count += accuracy
        mre_count += mre
        accuracy_count_norm += accuracy_norm
        mre_count_norm += mre_norm
        miss_count += miss
        count += 1

    # print('total accuracy across all graphs = ', accuracy_count/count)
    # print("mean relative error across all graphs :", mre_count/count)

    if count != 0:
        return accuracy_count / count, mre_count / count, accuracy_count_norm / count, mre_count_norm / count, miss_count / count, variance_actual_count / count, variance_predicted_count / count
    else:
        return Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid


def run_test(N, graph_rnd_list, density_list, output_url):
    for density in density_list:

        accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual, variance_predicted = test_model_multigraph(model, N, density, Radius, graph_rnd_list)
        print("N = {}, Density = {}".format(N, density))
        print("accuracy = {}, mre = {}".format(accuracy, mre))
        print("accuracy_norm = {}, mre_norm = {}".format(accuracy_norm, mre_norm))
        print("miss rate = {}".format(miss))
        print("SD of actual path stretch = {}".format(variance_actual ** 0.5))
        print("SD of predicted path stretch = {}".format(variance_predicted ** 0.5))

        f = open(output_url, 'a')  # open the csv file in the append mode
        writer = csv.writer(f)  # create the csv writer
        writer.writerow([dt_string, model_name, N, density, accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual**0.5, variance_predicted**0.5])
        f.close()

model = torch.jit.load(url)
model.eval()
print(model)

graph_rnd_list =  [39, 42, 43, 50, 52]
run_test(10, graph_rnd_list, density_list, output_url)

graph_rnd_list = [40, 41, 43, 50, 52]
run_test(25, graph_rnd_list, density_list, output_url)

graph_rnd_list = [6, 17, 18, 19, 21]
run_test(50, graph_rnd_list, density_list, output_url)

# graph_rnd_list = [0, 5, 6, 9, 13, 14, 20, 25, 27, 37, 41, 43, 50, 52, 58, 60, 63, 65, 68, 69]
graph_rnd_list = list(range(20))
run_test(27, graph_rnd_list, density_list, output_url)

# graph_rnd_list = [2, 13, 17, 18, 20, 24, 26, 27, 39, 40, 52, 53, 54, 60, 64, 65, 68, 70, 72, 85]
graph_rnd_list = list(range(20))
run_test(64, graph_rnd_list, density_list, output_url)

# graph_rnd_list = [6, 16, 32, 35, 36, 43, 44, 66, 67, 69, 79, 83, 85, 86, 89, 101, 106, 108, 109, 118]
graph_rnd_list = list(range(20))
run_test(125, graph_rnd_list, density_list, output_url)

# graph_rnd_list = [90, 97, 111, 142, 147, 165, 178, 230, 256, 311, 316, 341, 351, 376, 424, 434, 464, 472, 476, 511]
graph_rnd_list = list(range(20))
run_test(216, graph_rnd_list, density_list, output_url)

# graph_rnd_list = [25, 74, 106, 379, 405, 415, 621, 642, 701, 726, 729, 742, 762, 829, 900, 914, 973, 1084, 1093, 1107]
# graph_rnd_list = list(range(20))
# run_test(343, graph_rnd_list, density_list, output_url)

# graph_rnd_list = [54, 79, 108, 111, 258, 457, 521, 734, 829, 1314, 1585, 1669, 1834, 2017, 2370, 2379, 2468, 2582, 2923, 3187]
# graph_rnd_list = list(range(20))
# run_test(512, graph_rnd_list, density_list, output_url)