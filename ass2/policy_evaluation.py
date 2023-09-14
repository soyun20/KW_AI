# -*- coding: utf-8 -*-
"""Policy_evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1092aJ3ZXx_Zxvzq--jMQlnQ4vnuYi5hb
"""

#Policy Evaluation
import numpy as np

def get_state(state, action):
  action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)] #up, down, left, right

  state[0]+=action_grid[action][0] # add action
  state[1]+=action_grid[action][1]

  if state[0] < 0: # if out range of array
    state[0] = 0
  elif state[0] > 6:
    state[0] = 6

  if state[1] < 0:
    state[1] = 0
  elif state[1] > 6:
    state[1] = 6

  return state[0], state[1]

def policy_evaluation(grid_width, grid_height, action, policy, iter_num, reward=-1, dis=1):

  #table initialize
  post_value_table = np.zeros([grid_height, grid_width], dtype=float) #create array
  post_value_table = np.full([grid_height, grid_width], -1) #full -1
  post_value_table[0][0] = 0
  post_value_table[6][6] = 0
  post_value_table[0][2] = -100 #insert obstacle
  post_value_table[1][2] = -100
  post_value_table[3][4] = -100
  post_value_table[3][5] = -100
  post_value_table[6][2] = -100
  post_value_table[6][3] = -100

  #iteration
  if iter_num == 0:
    print('Iteration: {} \n{}\n'.format(iter_num, post_value_table))
    return post_value_table

  for iteration in range(iter_num):
    next_value_table = np.zeros([grid_height, grid_width], dtype=float)
    if iteration == 0:
        print('Iteration: {} \n{}\n'.format(iteration, post_value_table))
    for i in range(grid_height):
      for j in range(grid_width):
        if i == j and ((i==0) or (i==6)): # start, end point
          value_t = 0
        else:
          value_t = 0
          for act in action:
            i_, j_ = get_state([i,j], act) # get state
            value = policy[i][j][act] * (reward + dis*post_value_table[i_][j_]) # calculate value
            value_t += value
        next_value_table[i][j] = round(value_t, 3) #update value
    iteration +=1

    #print result
    if iteration < 10:
      print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
    if(iteration % 10) != iter_num:
      if iteration > 100:
        if(iteration % 20) == 0:
          print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
      else:
        if(iteration % 10) == 0:
          print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
    else:
      print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
    
    post_value_table = next_value_table
  
  return next_value_table

# 7 x 7 array
grid_width = 7
grid_height = grid_width
action = [0, 1, 2, 3] #up, down, left, right
policy = np.empty([grid_height, grid_width, len(action)], dtype=float) #create array

for i in range(grid_height):
  for j in range(grid_width):
    for k in range(len(action)):
      if i == j and ((i==0) or (i==6)):
        policy[i][j]=0.00 # start, end point
      else:
        policy[i][j]=0.25

policy[0][0] = [0] * 4
policy[6][6] = [0] * 4

value = policy_evaluation(grid_width, grid_height, action, policy, 1000)