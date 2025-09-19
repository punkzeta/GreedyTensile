import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from math import log
import pandas as pd
import csv
from datetime import datetime

#Basic PyTorch setup
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

#Use GPU if available
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_url='..\\model\\GreedyTensile_supervised_euclidean_n50_d5_rnd19_s3.pt'
output_url='test_result_euclidean_graphs.csv'

N = 50
Destination = 22
Density = 5
Radius = 1000
graph_rnd = 19
margin = 0.05
density_list = [2, 3, 4, 5]
num_of_sources = 3
len_of_relevance_grade = 3

initialization_rnd = 2000
permutation_rnd = initialization_rnd

now = datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("datetime = ", dt_string)

model_name = model_url.split('model')[-1][0:-3]
print(model_name)

l_con_min = 1.05
alpha = -4.4732
beta = 13.0715
def get_l_con(G, density, src, dst):

  delta =  get_dist(G, src, dst)/Radius

  if delta < 1:
    l_con = 1.
  else:
    l_con = max(1+ (alpha * math.log(delta)+ beta)/(density**2), l_con_min)

  return l_con

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

def get_src_dst(G):
  lst = np.random.choice(G.nodes, 2)
  return lst[0], lst[1]

def get_src_dst_1(G):
  dst = len(G.nodes)-1
  src = np.random.choice(list(range(0, len(G.nodes)-1)), 1)
  return src[0], dst

def get_degree(G, node):
  return len(G[node])

def get_ellipse_factor(G, src, dst, v):
  return (get_dist(G, src, v)+get_dist(G, v, dst))/get_dist(G, src, dst)

def get_cumulative_node_stretch_from_path(G, src, path):
  cumulative_node_stretch = 0.0
  dst = path[-1]
  for i in range(len(path)-1):
    cumulative_node_stretch += get_ellipse_factor(G, src, dst, path[i+1])

  return cumulative_node_stretch

def get_cumulative_d_ns_from_path(G, src, path):
  sum = 0.0
  dst = path[-1]
  for i in range(len(path)-1):
    sum += get_dist(G, path[i], path[i+1]) * get_ellipse_factor(G, src, dst, path[i+1])

  return sum

def get_max_path_epllispe_factor(G, src, dst, path):
  dist = get_dist(G, src, dst)
  max_ef = 1.

  for node in path:
    ef = get_ellipse_factor(G, src, dst, node)
    # print("Ellipse factor of node {} in path {}: {}".format(node, path, ef))
    if ef > max_ef:
      max_ef = ef

  return max_ef

def get_q_value(G, v, u, src, dst):
  sp = nx.dijkstra_path(G, v ,dst)
  path = nx.dijkstra_path(G, u ,dst)
  reward = get_dist(G, v, u)/Radius

  sp_length = get_path_length(G, sp)/Radius
  path_length = reward + get_path_length(G, path)/Radius

  path_stretch = get_path_length(G, nx.dijkstra_path(G, src ,dst))/get_dist(G, src, dst)

  penalty = path_stretch*(1.0+margin)*sp_length
  if path_length <= sp_length*path_stretch*(1.0+margin):
    q = path_length
  else:
    q = path_length + (path_length - path_stretch*(1.0+margin)*sp_length) * 2
  return q

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

  return G, dict(nx.all_pairs_bellman_ford_path(G)), dict(nx.all_pairs_bellman_ford_path_length(G))

def sort_ellipse_factor(G, dst, all_pairs_sp):

  dict_ef = dict()
  for src in G.nodes - {dst}:
    path = all_pairs_sp[src][dst]

    #sort the path by the maximum ellipse factor in the path
    dict_ef[src] = get_max_path_epllispe_factor(G, src, dst, all_pairs_sp[src][dst])


  print(dict_ef)
  dict_ef_sorted = dict(sorted(dict_ef.items(), key=lambda item: item[1]))
  print(dict_ef_sorted)

  return dict_ef_sorted

def get_sample_source_set(dict_ef_sorted, num_of_sources):
  key_list = list(dict_ef_sorted.keys())

  print("removed sources: {}".format(key_list[num_of_sources:]))
  return key_list[0:num_of_sources]

def get_excluded_source_set(dict_ef_sorted, num_of_sources):
  key_list = list(dict_ef_sorted.keys())

  print("removed sources: {}".format(key_list[num_of_sources:]))
  return key_list[num_of_sources:]

from sklearn.metrics import dcg_score
import math

def analyze_routing(G, src, dst, all_pairs_sp):

  ef_similarity_total = 0.0
  dist2dst_similarity_total = 0.0

  selected_path = list(all_pairs_sp[src][dst])
  count = len(selected_path)-1

  for v in selected_path[:-1]:
    # print("================= v = {} =================".format(v))
    dict_ef = dict()
    dict_dist2dst = dict()
    dict_q = dict()
    dict_q_rel = dict()

    # path = all_pairs_sp[v][dst]
    # print("selected forwarder to find the shortest path: {}".format(path[1]))
    for u in G[v]:
      dict_ef[u] = get_ellipse_factor(G, src, dst, u)
      dict_dist2dst[u] = get_dist(G, u, dst)/Radius
      dict_q[u] = get_q_value(G, v, u, src, dst)

    dict_q_sorted = dict(sorted(dict_q.items(), key=lambda item: item[1]))
    # print("sorted q values of {}'s neighbors: {}".format(v, dict_q_sorted))

    # [Discounted Cumulative Gain] calculate relevance grade for each neighbor of v according to its q-value
    for i in range(len(dict_q_sorted.keys())):
      # dict_q_rel[list(dict_q_sorted.keys())[i]] = 1/math.log2(2+i)
        dict_q_rel[list(dict_q_sorted.keys())[i]] = (len(dict_q_sorted.keys()) - i)**2

    # print("relavance grades of {}'s neighbors: {}".format(v, dict_q_rel))

    # [Discounted Cumulative Gain] calculate the baseline DCG value according to the key sequence in dict_q_sorted
    l = list(dict_q_sorted.keys())
    DCG_q = 0.0
    for i in range(len(dict_q_sorted.keys())):
      DCG_q += dict_q_rel[l[i]]/math.log2(i+2)
    # print("DCG_q = {}".format(DCG_q))

    dict_ef_sorted = dict(sorted(dict_ef.items(), key=lambda item: item[1]))
    # print("sorted ellipse factors of {}'s neighbors: {}".format(v, dict_ef_sorted))

    # [Discounted Cumulative Gain] calculate the DCG value according to the key sequence in dict_ef_sorted
    l = list(dict_ef_sorted.keys())
    DCG_ef = 0.0
    for i in range(len(dict_ef_sorted.keys())):
      DCG_ef += dict_q_rel[l[i]]/math.log2(i+2)
    # print("DCG_ef = {}".format(DCG_ef))
    ef_similarity_total += DCG_ef/DCG_q

    dict_dist2dst_sorted = dict(sorted(dict_dist2dst.items(), key=lambda item: item[1]))
    # print("sorted u-to-dst distances of {}'s neighbors: {}".format(v, dict_dist2dst_sorted))

    # [Discounted Cumulative Gain] calculate the DCG value according to the key sequence in dict_dist2dst_sorted
    l = list(dict_dist2dst_sorted.keys())
    DCG_dist = 0.0
    for i in range(len(dict_dist2dst_sorted.keys())):
      DCG_dist += dict_q_rel[l[i]]/math.log2(i+2)
    # print("DCG_dist = {}".format(DCG_dist))
    dist2dst_similarity_total += DCG_dist/DCG_q

  # print("average similarity between dict_q_sorted and dict_ef_sorted: {}".format(ef_similarity_total/count))
  # print("average similarity between dict_q_sorted and dict_dist2dst_sorted: {}".format(dist2dst_similarity_total/count))

  return ef_similarity_total/count, dist2dst_similarity_total/count

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

from torch.nn.functional import normalize
def shortest_path(model, G, src, dst):

  count = 0
  dummy = 1000
  v = src
  path = [v]
  visited = {v}
  done = False
  measure_similarity_total = 0.

  while done == False:
    q_values = np.zeros(len(G.nodes))
    q_values += dummy

    dict_measure = dict()
    dict_q = dict()
    dict_q_rel = dict()

    #Predict the q values of the device's neighbors
    for u in G[v]:
      if u not in visited:
        x_data = torch.Tensor(get_x_data(G, v, u, src, dst)).to(dev)
        # print("train x ({}, {}) = {}".format(v, u, x_data))
        x_data = normalize(x_data, p=2.0, dim = 0)
        # print("normalized train x ({}, {}) = {}".format(v, u, x_data))
        q_values[u] = model(x_data)

        dict_measure[u] = q_values[u]
        dict_q[u] = get_q_value(G, v, u, src, dst)

    count += 1
    # print("v = ", v)
    # print("q_values = ", q_values)
    #Determine the next node
    v = np.argmin(q_values)

    #Check if there is no available neighbor
    # if q_values[v] == dummy or count > 5:
    if q_values[v] == dummy or count >= len(G.nodes):
      done = True

    # print("select ", v)

    path.append(v)
    visited.add(v)

    # DCG calculation
    dict_q_sorted = dict(sorted(dict_q.items(), key=lambda item: item[1]))
    for i in range(len(dict_q_sorted.keys())):
      if i < len_of_relevance_grade:
        dict_q_rel[list(dict_q_sorted.keys())[i]] = (len(dict_q_sorted.keys()) - i)**2
      else:
        dict_q_rel[list(dict_q_sorted.keys())[i]] = 0.0

    l = list(dict_q_sorted.keys())
    DCG_q = 0.0
    # for i in range(len(dict_q_sorted.keys())):
    for i in range(min(len_of_relevance_grade, len(dict_q_sorted.keys()))):
      DCG_q += dict_q_rel[l[i]]/math.log2(i+2)

    dict_measure_sorted = dict(sorted(dict_measure.items(), key=lambda item: item[1]))
    l = list(dict_measure_sorted.keys())
    DCG_measure = 0.0
    # for i in range(len(dict_measure_sorted.keys())):
    for i in range(min(len_of_relevance_grade, len(dict_measure_sorted.keys()))):
      DCG_measure += dict_q_rel[l[i]]/math.log2(i+2)
    # print("DCG_dist = {}".format(DCG_dist))

    # print("DCG_measure={}, DCG_q={}".format(DCG_measure, DCG_q))
    if DCG_q != 0:
      measure_similarity_total += DCG_measure/DCG_q
    else:
      measure_similarity_total += 1

    # print("path = ", path)
    # print("visited = ", visited)

    if v == dst:
      done = True

  return path, measure_similarity_total/count

def generate_examples_2(G, excluded_list, dst):
  dataset_x = list()
  dataset_y = list()

  l = list(G.nodes)
  l.remove(dst)

  for node in excluded_list:
    l.remove(node)

  for src in l:
    for edge in G.edges:
      select = True
      for node in excluded_list:
        if node in edge:
          select = False
      if select == True:

        v = edge[0]
        u = edge[1]

        # print("(state, action) = (" + str(v) +", " + str(u) + ")")

        #Fill up state features of v
        s_0 = get_dist(G, src, dst)/Radius
        s_1 = get_ellipse_factor(G, src, dst, v)
        s_2 = get_dist(G, v, dst)/Radius

        #Fill up action features of u
        a_0 = get_dist(G, src, dst)/Radius
        a_1 = get_ellipse_factor(G, src, dst, u)
        a_2 = get_dist(G, u, dst)/Radius


        x_data = [s_0, s_1, s_2, a_0, a_1, a_2]
        y_data = [get_q_value(G, v, u, src, dst)]

        dataset_x.append(x_data)
        dataset_y.append(y_data)
        # print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

        v = edge[1]
        u = edge[0]
        s_0 = get_dist(G, src, dst)/Radius
        s_1 = get_ellipse_factor(G, src, dst, v)
        s_2 = get_dist(G, v, dst)/Radius
        a_0 = get_dist(G, src, dst)/Radius
        a_1 = get_ellipse_factor(G, src, dst, u)
        a_2 = get_dist(G, u, dst)/Radius
        x_data = [s_0, s_1, s_2, a_0, a_1, a_2]
        y_data = [get_q_value(G, v, u, src, dst)]
        dataset_x.append(x_data)
        dataset_y.append(y_data)
        # print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

  return(np.array(dataset_x), np.array(dataset_y))

def generate_examples_selected(G, src_list, dst, all_pairs_sp):
  dataset_x = list()
  dataset_y = list()


  print("src_list = ", src_list)
  for src in src_list:
    #Get nodes on the shortest path
    selected_list = all_pairs_sp[src][dst]

    print("selected_list = ", selected_list)

    for edge in G.edges:
      select = False
      for node in selected_list:
        if node in edge:
          select = True
      if select == True:
        v = edge[0]
        u = edge[1]

        # print("(state, action) = (" + str(v) +", " + str(u) + ")")

        #Fill up state features of v
        s_0 = get_dist(G, src, dst)/Radius
        s_1 = get_ellipse_factor(G, src, dst, v)
        s_2 = get_dist(G, v, dst)/Radius

        #Fill up action features of u
        a_0 = get_dist(G, src, dst)/Radius
        a_1 = get_ellipse_factor(G, src, dst, u)
        a_2 = get_dist(G, u, dst)/Radius


        x_data = [s_0, s_1, s_2, a_0, a_1, a_2]
        y_data = [get_q_value(G, v, u, src, dst)]

        dataset_x.append(x_data)
        dataset_y.append(y_data)
        print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

        v = edge[1]
        u = edge[0]
        s_0 = get_dist(G, src, dst)/Radius
        s_1 = get_ellipse_factor(G, src, dst, v)
        s_2 = get_dist(G, v, dst)/Radius
        a_0 = get_dist(G, src, dst)/Radius
        a_1 = get_ellipse_factor(G, src, dst, u)
        a_2 = get_dist(G, u, dst)/Radius
        x_data = [s_0, s_1, s_2, a_0, a_1, a_2]
        y_data = [get_q_value(G, v, u, src, dst)]
        dataset_x.append(x_data)
        dataset_y.append(y_data)
        print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

  return(np.array(dataset_x), np.array(dataset_y))

def get_random_sources(G, dst, k, rnd):
  random.seed(rnd)
  l = list(G.nodes)
  l.remove(dst)
  src_list = random.sample(l, k)

  return src_list

def generate_examples_selected_2(G, src_list, dst, all_pairs_sp):
  dataset_x = list()
  dataset_y = list()


  print("src_list = ", src_list)
  for src in src_list:
    #Get nodes on the shortest path
    selected_list = all_pairs_sp[src][dst]

    print("selected_list = ", selected_list)


    for v in selected_list:
      if v is not dst:
        for u in list(G[v]):
          # print("(state, action) = (" + str(v) +", " + str(u) + ")")
          #Fill up state features of v
          s_0 = get_dist(G, src, dst)/Radius
          s_1 = get_ellipse_factor(G, src, dst, v)
          s_2 = get_dist(G, v, dst)/Radius

          #Fill up action features of u
          a_0 = get_dist(G, src, dst)/Radius
          a_1 = get_ellipse_factor(G, src, dst, u)
          a_2 = get_dist(G, u, dst)/Radius


          x_data = [s_0, s_1, s_2, a_0, a_1, a_2]
          y_data = [get_q_value(G, v, u, src, dst)]

          dataset_x.append(x_data)
          dataset_y.append(y_data)
          print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

  return(np.array(dataset_x), np.array(dataset_y))

train_G, all_pairs_sp, all_pairs_sp_length = set_graph(N, Density, Radius, graph_rnd)

dict_ef_sorted = sort_ellipse_factor(train_G, Destination, all_pairs_sp)
# src_list = get_excluded_source_set(dict_ef_sorted, int((N-1) - 1))

key_list = list(dict_ef_sorted.keys())
candidate_list = key_list[:-num_of_sources]

#Sort the candidate list according to the difference between ef_similarity and dist_similarity
dict_similarity_diff = dict()
for i in range(len(candidate_list)):
  ef_similarity, dist_similarity = analyze_routing(train_G, candidate_list[i], Destination, all_pairs_sp)
  dict_similarity_diff[candidate_list[i]] = abs(dist_similarity - ef_similarity)

print("dict_similarity_diff = ", dict_similarity_diff)
dict_similarity_diff_sorted = dict(sorted(dict_similarity_diff.items(), key=lambda item: item[1]))
print("dict_similarity_diff_sorted = ", dict_similarity_diff_sorted)


key_list = list(dict_similarity_diff_sorted.keys())
src_list = key_list[0:num_of_sources]

# print("candidate_list = ", candidate_list)
# random.seed(1)
# src_list = random.sample(candidate_list, num_of_sources)

print("src = ", src_list)
# src_list = get_random_sources(train_G, Destination, num_of_sources, rnd=0)
X_train, Y_train = generate_examples_selected(train_G, src_list, Destination, all_pairs_sp)

X_train = torch.Tensor(X_train).to(dev)
Y_train = torch.Tensor(Y_train).to(dev)
print(X_train.shape)
print(Y_train.shape)

X_train = normalize(X_train, p=2.0, dim = 0)
# print(X_train)

# Y_train = normalize(Y_train, p=2.0, dim = 0)
# print(Y_train)

class Q_Network(nn.Module):
    def __init__(self, input_dim, hid1_dim, hid2_dim, output_dim, rnd):
        # initialze the superclass
        super(Q_Network, self).__init__()
        # input to first hidden layer
        torch.manual_seed(rnd)
        self.lin1 = nn.Linear(input_dim, hid1_dim)
        self.act1 = nn.LeakyReLU()
        # second hidden layer
        torch.manual_seed(rnd)
        self.lin2 = nn.Linear(hid1_dim, hid2_dim)
        self.act2 = nn.LeakyReLU()
        # third hidden layer and output
        torch.manual_seed(rnd)
        self.lin3 = nn.Linear(hid2_dim, output_dim)

    # this is where the meat of the action is
    def forward(self, x):
        # print("x0: ", x)
        x = self.lin1(x)
        # print(self.lin1.weight)
        # print("x1: ", x)
        x = self.act1(x)
        # print("x2: ", x)
        x = self.lin2(x)
        # print("x3: ", x)
        x = self.act2(x)
        # print("x4: ", x)
        x = self.lin3(x)
        # print("x5: ", x)
        return x

# def weights_init(model, rnd):
#     torch.manual_seed(rnd)
#     for m in model.modules():
#         if isinstance(m, nn.Linear):
#             # initialize the weight tensor, here we use a normal distribution
#             m.weight.data.normal_(0, 1)

import math
import numpy
#SGD w/ mini-batches
#-------------------------------
#But as you can see, if you give batch training a try, it takes longer to converge and
#isn't as reliable. Turns out noise realy is beneficial.
#The modern approach is to split the difference between the two. Use more than one data
#point at a time for training, but only a small random selection from the dataset: a mini-batch

def train_model(model, X, Y, mini_batch_size=50):
  loss_func = nn.MSELoss().to(dev)
  # loss_func = nn.L1Loss().to(dev)
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

  epochs = 5000
  N = X.size(0)
  for i in range(epochs):
    np.random.seed(permutation_rnd + i)
    order=np.random.permutation(N) #Generate a random ordering (this ensures every datapoint is used for training, just in random order
    # order=list(range(N))
    for j in range(0,N,mini_batch_size):
      data_points = order[j:min(j+mini_batch_size,N)]
      x_var = Variable(X[data_points], requires_grad=True)
      y_var = Variable(Y[data_points], requires_grad=False)

      optimizer.zero_grad() #Zero the gradient accumulators (individual gradient values can come from many sources, remember our discussion on re-using weights, we sum them all)
      y_hat = model(x_var) #Calculate the output of the model
      # if i == 0:
      #   print("nan occur: ", numpy.isnan(y_hat.detach().numpy()).any())

      loss = loss_func.forward(y_hat, y_var) #Calculate loss between model output and target values
      loss.backward() #Back propogation - calcluate gradients by propogating loss backwards through the network
      # for param in model.parameters():
      #   print(param.grad)
      # clipping_value = 1
      # nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
      optimizer.step() #Use the gradients to update weights, take a "step" in the loss-minimizing direction

    if i % 2000 == 0:
      print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.cpu().numpy()))
      assert math.isnan(loss.data.cpu().numpy()) != True

# evaluate the model
def evaluate_model(model, X, Y):
  # print('model = ', model)
  predictions, actuals = list(), list()
  N = X.size(0)

  y_hat = model(X)
  # print(y_hat)
  y_hat = y_hat.detach().numpy()
  actual = Y.numpy()
  predictions.append(y_hat)
  actuals.append(actual)

  predictions, actuals = np.vstack(predictions), np.vstack(actuals)

  return predictions, actuals

def test_model_single(model, G, dst, all_pairs_sp, all_pairs_sp_length):
  accuracy_count = 0
  count = 0
  mre = 0.0
  accuracy_count_norm = 0
  mre_norm = 0
  miss = 0
  path_DCG_total = 0

  l = list(G.nodes)
  l.remove(dst)

  for src in l:
    s_path = all_pairs_sp[src][dst]
    # print('Dijkstra shortest path = ', s_path)
    actual_dist = all_pairs_sp_length[src][dst]/Radius
    # print('Dijkstra shortest path distance = ', actual_dist)

    path, path_DCG = shortest_path(model, G, src, dst)
    # print('predicted shortest path = ', path)
    predict_dist = get_path_length(G, path)/Radius
    # print('predicted shortest path distance = ', predict_dist)

    path_stretch = actual_dist/(get_dist(G, src, dst)/Radius)
    # print("path_stretch = ", path_stretch)

    count += 1

    if dst not in path:
      path = []
      predict_dist = 0
      mre += 1.0
      mre_norm += 1.0
      miss += 1
    else:
      path_DCG_total += path_DCG

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
  # print('accuracy = ', accuracy_count/count)
  # print("mre:", mre/count)

  return accuracy_count/count, mre/count, accuracy_count_norm/count, mre_norm/count, miss/count, path_DCG_total/count

def test_model(model, G, all_pairs_sp, all_pairs_sp_length):
  accuracy_count = 0.0
  mre_count = 0.0
  accuracy_count_norm = 0
  mre_count_norm = 0
  miss_count = 0
  count = 0
  path_DCG_total = 0

  for dst in G.nodes:
    accuracy, mre, accuracy_norm, mre_norm, miss, path_DCG = test_model_single(model, G, dst, all_pairs_sp, all_pairs_sp_length)
    accuracy_count += accuracy
    mre_count += mre
    accuracy_count_norm += accuracy_norm
    mre_count_norm += mre_norm
    miss_count += miss
    path_DCG_total += path_DCG
    count += 1

  # print('total accuracy = ', accuracy_count/count)
  # print("mean relative error:", mre_count/count)
  return accuracy_count/count, mre_count/count, accuracy_count_norm/count, mre_count_norm/count, miss_count/count, path_DCG_total/count

def test_model_multigraph(model, size, density, radius, rnd_list):
  accuracy_count = 0.0
  mre_count = 0.0
  accuracy_count_norm = 0
  mre_count_norm = 0
  miss_count = 0
  count = 0
  path_DCG_total = 0

  for rnd in rnd_list:
    G, all_pairs_sp, all_pairs_sp_length = set_graph(size, density, radius, rnd)
    accuracy, mre, accuracy_norm, mre_norm, miss, path_DCG = test_model(model, G, all_pairs_sp, all_pairs_sp_length)
    accuracy_count += accuracy
    mre_count += mre
    accuracy_count_norm += accuracy_norm
    mre_count_norm += mre_norm
    miss_count += miss
    path_DCG_total += path_DCG
    count += 1

  # print('total accuracy across all graphs = ', accuracy_count/count)
  # print("mean relative error across all graphs :", mre_count/count)

  return accuracy_count/count, mre_count/count, accuracy_count_norm/count, mre_count_norm/count, miss_count/count, path_DCG_total/count

def run_test(N, graph_rnd_list, density_list):
  for density in density_list:
    f = open(output_url, 'a') #open the csv file in the append mode
    writer = csv.writer(f) #create the csv writer

    accuracy, mre, accuracy_norm, mre_norm, miss, path_DCG = test_model_multigraph(model, N, density, Radius, graph_rnd_list)
    print("N = {}, Density = {}".format(N, density))
    print("accuracy = {}, mre = {}".format(accuracy, mre))
    print("accuracy_norm = {}, mre_norm = {}".format(accuracy_norm, mre_norm))
    print("miss rate = {}".format(miss))
    print("avg DCG = {}".format(path_DCG))

    writer.writerow([dt_string, model_name, N, density, initialization_rnd, src_list, accuracy, mre, accuracy_norm, mre_norm, miss])
    f.close()

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

done = False
print(X_train.size(0))


while(done == False):
  print("initialization_rnd = {}".format(initialization_rnd))
  model = Q_Network(6, N*2*6, 6, 1, initialization_rnd).to(dev)

  try:
    train_model(model, X_train, Y_train)
    predictions, actuals = evaluate_model(model, X_train, Y_train)
    print('MSE: %.8f' % mean_squared_error(actuals, predictions))

    accuracy, mre, accuracy_norm, mre_norm, miss, path_DCG = test_model(model, train_G, all_pairs_sp, all_pairs_sp_length)
    print("accuracy_norm = {}".format(accuracy_norm))

    assert accuracy_norm > 0.95

    done = True

  except:
    initialization_rnd += 1

s_path = nx.dijkstra_path(train_G, 0 ,Destination)
print('Dijkstra shortest path = ', s_path)
actual_dist = get_path_length(train_G, s_path)
print('Dijkstra shortest path distance = ', actual_dist)
path, path_DCG = shortest_path(model, train_G, 0, Destination)
print("Predicted path = {}, path_DCG = {}".format(path, path_DCG))

test_model_single(model, train_G, Destination, all_pairs_sp, all_pairs_sp_length)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save(model_url) # Save

graph_rnd_list = [39, 42, 43, 50, 52]
run_test(10, graph_rnd_list, density_list)

graph_rnd_list = [40, 41, 43, 50, 52]
run_test(25, graph_rnd_list, density_list)

graph_rnd_list = [6, 17, 18, 19, 21]
run_test(50, graph_rnd_list, density_list)