import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import math
from math import log
import pandas as pd
import csv
from datetime import datetime
from scipy import stats
import itertools

url='..\\model\\GreedyTensile_supervised_hyperbolic_n50_d1_rnd19_s3.pt'
output_url='test_result_hyperbolic_graphs.csv'

N = 50
Destination = 22
Degree = 1
Curvature = .6
graph_rnd = 19
margin = 0.05
degree_list = [1, 2, 3, 4]
sample_all_node = False
num_of_sources = 3
Invalid = -1
min_d = 1.
min_sf = 1.

initialization_rnd = 2000
permutation_rnd = initialization_rnd

now = datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("datetime = ", dt_string)

model_name = url.split('/content/gdrive/My Drive/model/')[-1][0:-3]
print(model_name)

from numpy.random import randint
class Distribution :

    n = 10000 # standard random numbers to generate

    '''
    Constructor for this Distribution class.

    Args:
            dist (scipy.stats random variable): A random variable from the scipy stats libary.

    Attributes:
            dist (scipy.stats random variable): A random variable from the scipy stats libary.
            n (int): a number indicating how many random numbers should be generated in one batch
            randomNumbers: a list of n random numbers generated from 'dist'
            idx (int): a number keeping track of how many random numbers have been sampled

    '''

    def __init__(self, dist, rnd):
        self.dist = dist
        self.rnd_seed = rnd
        self.resample()
        # print("dist = ", dist)

    def __str__(self):
        return str(self.dist)

    def resample(self):
        rnd = self.rnd_seed
        self.randomNumbers = self.dist.rvs(self.n, random_state=np.random.RandomState(seed=self.rnd_seed))
        self.idx = 0

    def rvs(self, n=1):
        '''
        A function that returns n (=1 by default) random numbers from the specified distribution.

        Returns:
            One random number (float) if n=1, and a list of n random numbers otherwise.
        '''
        if self.idx >= self.n - n :
            while n > self.n :
                self.n *= 10
            self.resample()
        if n == 1 :
            rs = self.randomNumbers[self.idx]
        else :
            rs = self.randomNumbers[self.idx:(self.idx+n)]
        self.idx += n
        return rs

    def mean(self):
        return self.dist.mean()

    def std(self):
        return self.dist.std()

    def var(self):
        return self.dist.var()

    def cdf(self, x):
        return self.dist.cdf(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def sf(self, x):
        return self.dist.sf(x)

    def ppf(self, x):
        return self.dist.ppf(x)

    def moment(self, n):
        return self.dist.moment(n)

    def median(self):
        return self.dist.median()

    def interval(self, alpha):
        return self.dist.interval(alpha)

# np.random.RandomState(seed=graph_rnd)
# uniformSampler = Distribution(stats.uniform(), graph_rnd)

def invDensityRadius(y, a, R):
  return np.arcsinh((y*(np.cosh(a*R)-1))/a) / a

def sampleRadius(uniformSampler, a, R):
  return invDensityRadius(uniformSampler.rvs(), a, R)

def sampleAngle(uniformSampler):
  return uniformSampler.rvs() * 2* np.pi

def calculateDistance(p1, p2):
  #p1= (r, angle), p2 = (r', angle')
  dist_ang = np.pi - abs(np.pi - abs(p1[1]-p2[1]))

  return np.arccosh(np.cosh(p1[0])*np.cosh(p2[0]) - np.sinh(p1[0])*np.sinh(p2[0])* np.cos(dist_ang))

def get_two_nodes_dist(G, G_pos, node1, node2):
  p1 = G_pos[node1]
  p2 = G_pos[node2]
  return calculateDistance(p1, p2)

def get_path_length(G, G_pos, path):
  path_length = 0.0
  for i in range(len(path)-1):
    path_length += get_two_nodes_dist(G, G_pos, path[i], path[i+1])
  return path_length

def convertPolarToEuclidian(p):
  #p= (r, angle)
  x = p[0]*np.cos(p[1])
  y = p[0]*np.sin(p[1])
  return (x, y)

def get_degree(G, node):
  return len(G[node])

def get_stretch_factor(G, G_pos, src, dst, node):
  if src == dst:
    return 1.0
  return (get_two_nodes_dist(G, G_pos, src, node)+get_two_nodes_dist(G, G_pos, node, dst))/get_two_nodes_dist(G, G_pos, src, dst)

# def get_q_value(G, G_pos, all_pairs_sp, v, u, dst):
#   path = all_pairs_sp[u][dst]
#   reward = get_two_nodes_dist(G, G_pos, v, u)
#   q = get_path_length(G,G_pos, path)
#   return reward + q

def get_q_value(G, G_pos, all_pairs_sp, v, u, src, dst):
  sp = all_pairs_sp[v][dst]
  path = all_pairs_sp[u][dst]
  reward = get_two_nodes_dist(G, G_pos, v, u)
  dist = get_two_nodes_dist(G, G_pos, u, dst)

  sp_length = get_path_length(G, G_pos, sp)
  path_length = reward + get_path_length(G, G_pos, path)

  path_stretch = get_path_length(G, G_pos, all_pairs_sp[src][dst]) / get_two_nodes_dist(G, G_pos, src, dst)

  penalty = path_stretch*(1.0+margin)*sp_length
  if path_length <= sp_length*path_stretch*(1.0+margin) and dist <= get_two_nodes_dist(G, G_pos, v, dst):
    q = sp_length
  else:
    q = sp_length + (path_length - path_stretch*(1.0+margin)*sp_length) * 2
  return q

def get_rnd_number():
  return random.uniform(0, 1)

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

def generateGraph(uniformSampler, n, average_degrees, negative_curvature):
  v = average_degrees
  a = negative_curvature
  R = 2*np.log(n/v)

  pointsPositions = {}
  for i in range(n):
    pointsPositions[i] = (sampleRadius(uniformSampler, a, R), sampleAngle(uniformSampler))


  G = nx.Graph()
  # add all nodes
  for i in pointsPositions.keys():
    G.add_node(i)
  # add all edges between nodes if distance is smaller then radius
  for pair in itertools.combinations(pointsPositions.keys(), r=2):
    # print("pair = ", pair)
    p1 = pointsPositions[pair[0]]
    p2 = pointsPositions[pair[1]]
    if calculateDistance(p1, p2) <= R:
      G.add_edge(pair[0], pair[1], weight=calculateDistance(p1, p2))

  #get the positions of all nodes in euclidian coordinates
  layout = {k : convertPolarToEuclidian(pointsPositions[k]) for k in pointsPositions.keys()}

  # print("layout = ", layout)

  # nx.draw_networkx_nodes(G, pos=layout)
  # nx.draw_networkx_edges(G, pos=layout)
  # nx.draw_networkx_labels(G, pos=layout)
  # plt.title("Hyperbolic graph with size = {}".format(n))
  # plt.show()

  # degree_sequence = sorted((d for Size, d in G.degree()), reverse=True)
  # dmax = max(degree_sequence)
  # plt.title("Degree Distribution")
  # plt.ylabel("Degree")
  # plt.xlabel("Node Count")
  # plt.plot(degree_sequence, "b-", marker=".")
  # plt.show()

  all_pairs_sp, all_pairs_sp_length = dijkstra_all_pairs_shortest_path(G)

  return G, pointsPositions, all_pairs_sp, all_pairs_sp_length

def get_x_data(G, G_pos, v, u, src, dst):
  #Fill up state features of v
  s_0 = get_two_nodes_dist(G, G_pos, v, dst)
  s_1 = get_stretch_factor(G, G_pos, src, dst, v)-1.
  if math.isnan(s_1) == True:
    s_1 = 0.

  #Fill up action features of u
  a_0 = get_two_nodes_dist(G, G_pos, u, dst)
  a_1 = get_stretch_factor(G, G_pos, src, dst, u)-1.
  if math.isnan(a_1) == True:
    a_1 = 0.

  return np.array([s_0, s_1, a_0, a_1])

from torch.nn.functional import normalize
def shortest_path(model, G, G_pos, src, dst, all_pairs_sp, all_pairs_sp_length):

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
    #Predict the q values of the device's neighbors
    for u in G[v]:
      if u not in visited:
        x_data = torch.Tensor(get_x_data(G, G_pos, v, u, src, dst)).to(dev)
        # print("train x ({}, {}) = {}".format(v, u, x_data))
        x_data = normalize(x_data, p=2.0, dim = 0)
        # print("normalized train x ({}, {}) = {}".format(v, u, x_data))
        q_values[u] = model(x_data)

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

    # print("path = ", path)
    # print("visited = ", visited)

    if v == dst:
      done = True

  return path

def sort_path_stretch(G, G_pos, dst, all_pairs_sp, all_pairs_sp_length):

  dict_sf = dict()
  for src in G.nodes - {dst}:
    path = all_pairs_sp[src][dst]

    if len(all_pairs_sp[src][dst]) != 0:
      dict_sf[src] = all_pairs_sp_length[src][dst]/get_two_nodes_dist(G, G_pos, src, dst)


  print(dict_sf)
  dict_sf_sorted = dict(sorted(dict_sf.items(), key=lambda item: item[1]))
  print(dict_sf_sorted)

  return dict_sf_sorted

def generate_sanity_dataset(G, G_pos, dst, all_pairs_sp):
  dataset_x = list()
  dataset_y = list()

  l = list(G.nodes)
  l.remove(dst)
  for src in l:
  # if 1:
    # print("src, dst =", src,dst)
    for edge in G.edges:
      # if edge in all_pairs_sp[src][dst]:

        v = edge[0]
        u = edge[1]

        # print("(state, action) = (" + str(v) +", " + str(u) + ")")

        #Fill up state features of v
        s_0 = get_two_nodes_dist(G, G_pos, v, dst)
        # s_1 = get_degree(G, v)
        s_1 = get_stretch_factor(G, G_pos, src, dst, v)
        # rnd = get_rnd_number()
        # s_1 = rnd

        #Fill up action features of u
        a_0 = get_two_nodes_dist(G, G_pos, u, dst)
        # a_1 = get_degree(G, u)
        a_1 = get_stretch_factor(G, G_pos, src, dst, u)
        # rnd = get_rnd_number()
        # a_1 = rnd

        x_data = [s_0, s_1, a_0, a_1]
        y_data = [get_q_value(G, G_pos, all_pairs_sp, v, u, src, dst)]

        sample_flag = True
        for i in range(len(x_data)):
          if np.isnan(np.float64(x_data[i])) == True:
            sample_flag = False
            # print("v={}, u={}, x_data[{}] = {}".format(v, u, i, x_data[i]))
            break
        if np.isnan(np.float64(y_data[0])) == True:
          sample_flag = False
          # print("v={}, u={}, y_data = {}".format(v, u, i, y_data[0]))

        if sample_flag == True:
          dataset_x.append(x_data)
          dataset_y.append(y_data)
          print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

        v = edge[1]
        u = edge[0]
        sample_flag = True

        s_0 = get_two_nodes_dist(G, G_pos, v, dst)
        # s_1 = get_degree(G, v)
        s_1 = get_stretch_factor(G, G_pos, src, dst, v)
        # rnd = get_rnd_number()
        # s_1 = rnd

        a_0 = get_two_nodes_dist(G, G_pos, u, dst)
        # a_1 = get_degree(G, u)
        a_1 = get_stretch_factor(G, G_pos, src, dst, u)
        # rnd = get_rnd_number()
        # a_1 = rnd
        x_data = [s_0, s_1, a_0, a_1]
        y_data = [get_q_value(G, G_pos, all_pairs_sp, v, u, src, dst)]

        sample_flag = True
        for i in range(len(x_data)):
          if np.isnan(np.float64(x_data[i])) == True:
            sample_flag = False
            # print("v={}, u={}, x_data[{}] = {}".format(v, u, i, x_data[i]))
            break
        if np.isnan(np.float64(y_data[0])) == True:
          sample_flag = False
          # print("v={}, u={}, y_data = {}".format(v, u, i, y_data[0]))

        if sample_flag == True:
          dataset_x.append(x_data)
          dataset_y.append(y_data)
          print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

  return(np.array(dataset_x), np.array(dataset_y))

def generate_examples_selected(G, G_pos, src_list, dst, all_pairs_sp):
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
        s_0 = get_two_nodes_dist(G, G_pos, v, dst)
        # s_1 = get_degree(G, v)
        s_1 = get_stretch_factor(G, G_pos, src, dst, v)
        # rnd = get_rnd_number()
        # s_1 = rnd

        #Fill up action features of u
        a_0 = get_two_nodes_dist(G, G_pos, u, dst)
        # a_1 = get_degree(G, u)
        a_1 = get_stretch_factor(G, G_pos, src, dst, u)
        # rnd = get_rnd_number()
        # a_1 = rnd

        x_data = [s_0, s_1, a_0, a_1]
        y_data = [get_q_value(G, G_pos, all_pairs_sp, v, u, src, dst)]

        sample_flag = True
        for i in range(len(x_data)):
          if np.isnan(np.float64(x_data[i])) == True:
            sample_flag = False
            # print("v={}, u={}, x_data[{}] = {}".format(v, u, i, x_data[i]))
            break
        if np.isnan(np.float64(y_data[0])) == True:
          sample_flag = False
          # print("v={}, u={}, y_data = {}".format(v, u, i, y_data[0]))

        if sample_flag == True:
          dataset_x.append(x_data)
          dataset_y.append(y_data)
          print("train x ({}, {}, {}) = {}; y = {}".format(src, v, u, x_data, y_data))

        v = edge[1]
        u = edge[0]
        sample_flag = True
        s_0 = get_two_nodes_dist(G, G_pos, v, dst)
        # s_1 = get_degree(G, v)
        s_1 = get_stretch_factor(G, G_pos, src, dst, v)
        # rnd = get_rnd_number()
        # s_1 = rnd

        a_0 = get_two_nodes_dist(G, G_pos, u, dst)
        # a_1 = get_degree(G, u)
        a_1 = get_stretch_factor(G, G_pos, src, dst, u)
        # rnd = get_rnd_number()
        # a_1 = rnd
        x_data = [s_0, s_1, a_0, a_1]
        y_data = [get_q_value(G, G_pos, all_pairs_sp, v, u, src, dst)]

        sample_flag = True
        for i in range(len(x_data)):
          if np.isnan(np.float64(x_data[i])) == True:
            sample_flag = False
            # print("v={}, u={}, x_data[{}] = {}".format(v, u, i, x_data[i]))
            break
        if np.isnan(np.float64(y_data[0])) == True:
          sample_flag = False
          # print("v={}, u={}, y_data = {}".format(v, u, i, y_data[0]))

        if sample_flag == True:
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

# Commented out IPython magic to ensure Python compatibility.
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

uniformSampler = Distribution(stats.uniform(), graph_rnd)
train_G, train_G_pos, all_pairs_sp, all_pairs_sp_length = generateGraph(uniformSampler, N, Degree, Curvature)

if sample_all_node == True:
  X_train, Y_train = generate_sanity_dataset(train_G, train_G_pos, Destination, all_pairs_sp)
else:
  dict_sf_sorted = sort_path_stretch(train_G, train_G_pos, Destination, all_pairs_sp, all_pairs_sp_length)
  all_src_list = list(dict_sf_sorted.keys())
  src_list = all_src_list[:num_of_sources]

  X_train, Y_train = generate_examples_selected(train_G, train_G_pos, src_list, Destination, all_pairs_sp)

X_train = torch.Tensor(X_train).to(dev)
Y_train = torch.Tensor(Y_train).to(dev)
print(X_train.shape)
print(Y_train.shape)


X_train = normalize(X_train, p=2.0, dim = 0)
# print(X_train)

Y_train = normalize(Y_train, p=2.0, dim = 0)
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

def train_model(model, X, Y, mini_batch_size=10):
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

def test_model_single(model, G, G_pos, dst, all_pairs_sp, all_pairs_sp_length):
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

    if len(s_path) !=0:

      # print('Dijkstra shortest path = ', s_path)
      actual_dist = all_pairs_sp_length[src][dst]
      # print('Dijkstra shortest path distance = ', actual_dist)

      path = shortest_path(model, G, G_pos, src, dst, all_pairs_sp, all_pairs_sp_length)
      # print('predicted shortest path = ', path)
      predict_dist = get_path_length(G, G_pos, path)
      # print('predicted shortest path distance = ', predict_dist)

      path_stretch = actual_dist/(get_two_nodes_dist(G, G_pos, src, dst))
      # print("path_stretch = ", path_stretch)

      predicted_path_stretch = predict_dist/(get_two_nodes_dist(G, G_pos, src, dst))

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

        variance_actual += path_stretch**2
        variance_predicted += predicted_path_stretch**2

  if count !=0:
    return accuracy_count/count, mre/count, accuracy_count_norm/count, mre_norm/count, miss/count, variance_actual/count, variance_predicted/count
  else:
    return Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid

def test_model(model, G, G_pos, all_pairs_sp, all_pairs_sp_length):
  accuracy_count = 0.0
  mre_count = 0.0
  accuracy_count_norm = 0
  mre_count_norm = 0
  miss_count = 0
  variance_actual_count = 0
  variance_predicted_count = 0
  count = 0

  for dst in G.nodes:
    accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual, variance_predicted = test_model_single(model, G, G_pos, dst, all_pairs_sp, all_pairs_sp_length)
    if accuracy != Invalid:
      accuracy_count += accuracy
      mre_count += mre
      accuracy_count_norm += accuracy_norm
      mre_count_norm += mre_norm
      miss_count += miss
      variance_actual_count += variance_actual
      variance_predicted_count += variance_predicted
      count += 1

  # print('total accuracy = ', accuracy_count/count)
  # print("mean relative error:", mre_count/count)
  if count != 0:
    return accuracy_count/count, mre_count/count, accuracy_count_norm/count, mre_count_norm/count, miss_count/count, variance_actual_count/count, variance_predicted_count/count
  else:
    return Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid

def test_model_multigraph(model, size, degree, curvature, rnd_list):
  accuracy_count = 0.0
  mre_count = 0.0
  accuracy_count_norm = 0
  mre_count_norm = 0
  miss_count = 0
  variance_actual_count = 0
  variance_predicted_count = 0
  count = 0

  for rnd in rnd_list:
    uniformSampler = Distribution(stats.uniform(), rnd)
    G, G_pos, all_pairs_sp, all_pairs_sp_length = generateGraph(uniformSampler, size, degree, curvature)
    accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual, variance_predicted = test_model(model, G, G_pos, all_pairs_sp, all_pairs_sp_length)
    if accuracy != Invalid:
      accuracy_count += accuracy
      mre_count += mre
      accuracy_count_norm += accuracy_norm
      mre_count_norm += mre_norm
      miss_count += miss
      variance_actual_count += variance_actual
      variance_predicted_count += variance_predicted
      count += 1

  # print('total accuracy across all graphs = ', accuracy_count/count)
  # print("mean relative error across all graphs :", mre_count/count)
  if count != 0:
    return accuracy_count/count, mre_count/count, accuracy_count_norm/count, mre_count_norm/count, miss_count/count, variance_actual_count/count, variance_predicted_count/count
  else:
    return Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid

def run_test(N, graph_rnd_list, degree_list, output_url):
  for degree in degree_list:

    accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual, variance_predicted = test_model_multigraph(model, N, degree, Curvature, graph_rnd_list)
    print("N = {}, Degree = {}".format(N, degree))
    print("accuracy = {}, mre = {}".format(accuracy, mre))
    print("accuracy_norm = {}, mre_norm = {}".format(accuracy_norm, mre_norm))
    print("miss rate = {}".format(miss))
    print("SD of actual path stretch = {}".format(variance_actual**0.5))
    print("SD of predicted path stretch = {}".format(variance_predicted**0.5))

    f = open(output_url, 'a') #open the csv file in the append mode
    writer = csv.writer(f) #create the csv writer
    writer.writerow([dt_string, model_name, N, degree, accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual**0.5, variance_predicted**0.5])
    f.close()

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

done = False
print(X_train.size(0))


while(done == False):
# if 1:
  print("initialization_rnd = {}".format(initialization_rnd))
  model = Q_Network(4, N*2*4, 4, 1, initialization_rnd).to(dev)

  try:
  # if 1:
    train_model(model, X_train, Y_train)
    predictions, actuals = evaluate_model(model, X_train, Y_train)
    print('MSE: %.8f' % mean_squared_error(actuals, predictions))

    accuracy, mre, accuracy_norm, mre_norm, miss, variance_actual, variance_predicted = test_model(model, train_G, train_G_pos, all_pairs_sp, all_pairs_sp_length)
    print("accuracy_norm = {}".format(accuracy_norm))

    assert accuracy_norm > 0.97

    done = True

  except:
    initialization_rnd += 1

print('Dijkstra shortest path = ', all_pairs_sp[0][Destination])
print('Dijkstra shortest path distance = ', all_pairs_sp_length[0][Destination])
path = shortest_path(model, train_G, train_G_pos, 0, Destination, all_pairs_sp, all_pairs_sp_length)
print("Predicted path = ", path)
print("Predicted path length = ", get_path_length(train_G, train_G_pos, path))

test_model_single(model, train_G, train_G_pos, Destination, all_pairs_sp, all_pairs_sp_length)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save(url) # Save

graph_rnd_list = [39, 42, 43, 50, 52]
run_test(10, graph_rnd_list, degree_list, output_url)

graph_rnd_list = [40, 41, 43, 50, 52]
run_test(25, graph_rnd_list, degree_list, output_url)

graph_rnd_list = [6, 17, 18, 19, 21]
run_test(50, graph_rnd_list, degree_list, output_url)

graph_rnd_list = [6, 16, 32, 35, 36]
run_test(100, graph_rnd_list, degree_list, output_url)