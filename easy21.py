#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:25:31 2018

@author: Hinrik
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def getQ():
    return Q

def max_dict(q):

  max_key = None
  max_val = float('-inf')
 
  for k, v in q.items():
    if v == max_val:
        # if value is the same randomly pick action (stick or hit)
        max_key = np.random.choice([max_key, k])
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

# cards[0-9] are Red
# cards[10-19] are black
# 1/3 probability drawing red
def hit():
    if(np.random.randint(3) == 2):
        return cards[(np.random.randint(10))]
    return cards[(np.random.randint(10))+10]
    
def begin():
    return cards[(np.random.randint(10))+10]

# finishing episode - receiving reward
def game21(s):
    dealer_sum = s[0]
    while dealer_sum < 17:
        dealer_sum += hit()
        #if dealer goes bust
        if dealer_sum < 1 or dealer_sum > 21:
            return 1
    
    if dealer_sum > s[1]:
        return -1
    elif s[1] > dealer_sum:
        return 1
    else:
        return 0

#------------------------------
def epsilonGreedy(s):
    Nstate[state.index(s)] += 1
    epsilon = Nconst/(Nconst+Nstate[state.index(s)])
    p = np.random.random()
    if p < epsilon: # lets explore
        a = np.random.choice(action)
    else: # lets exploit
        a = max_dict(Q[s])[0] 
    return a

def step(s,a):
    if a == action[0]: # hit
        s = ((s[0], s[1]+hit()))
        if s not in state:
            return None, -1 # burst
        else:
            return s, 0 # not end of episode
        
    return  None,game21(s) # stick

def trainSarsa(s):
    a = epsilonGreedy(s)
    states_actions = [(s,a)]
    N[state.index(s)][action.index(a)] += 1
    game_over = False
    r = 0
    e = inite()
    while(s in state and game_over == False):
        next_state,r = step(s,a)
        
        if pd.isnull(next_state): # end of an episode
            game_over = True
            delta = r - Q[s][a]
        else:    
            next_a = epsilonGreedy(next_state)
            delta = r+Q[next_state][next_a] - Q[s][a]
        e[s][a] += 1
        
        for s,a in states_actions:
            alpha = 1/N[state.index(s)][action.index(a)]
            Q[s][a] += alpha*delta*e[s][a]
            e[s][a] = e[s][a]*lda
        if not game_over:
            s = next_state
            a = next_a
            N[state.index(s)][action.index(a)] += 1
            states_actions.append((s,a))
            
def trainMonteCarlo(s):

    a = epsilonGreedy(s)
    N[state.index(s)][action.index(a)] += 1
    
    #returns list containing all states and action taken episode except first
    returns = rollout(np.copy(s),np.copy(a))
    alpha = 1/N[state.index(s)][action.index(a)]
    
    Q[s][a] += ((returns[0]-Q[s][a])/Nstate[state.index(s)])
    #updating states action for all states actions this epesode except first
    for sa in returns[1]:
        N[state.index(sa[0])][action.index(sa[1])] += 1
        alpha = 1/N[state.index(sa[0])][action.index(sa[1])]
        Q[sa[0]][sa[1]] += (alpha * (returns[0]-Q[sa[0]][sa[1]]))
                    
def rollout(s,a):

    sa = list()
    while(True):
        s,r = step(s,a)
        if pd.isnull(s):
            return r,sa
        a = epsilonGreedy(s)
        sa.append((s,a))
        

def init():
    Nstate = np.zeros(len(state))    
    N = np.zeros(shape=(len(state),2))
    Q = {}
    e = {}
    for s in state:
        Q[s] = {}
        e[s] = {}
        for a in action:
            Q[s][a] = 0
            e[s][a] = 0
    return Nstate,N,Q

def inite():
    e = {}
    for s in state:
        e[s] = {}
        for a in action:
            e[s][a] = 0
    return e 
#------------------------

Nconst = 100
cards = list(range(-10,0)) + list(range(1,11))
state = list()
action = ['h', 's']
dealers_min_top_choice = 17

for dealer in range(1,11):
    for player in range (1,22):
        state.append((dealer,player))
        
Nstate,N,Q = init()
#------------------------
# train monte carlo and create V*
#------------------------
episodes = 100000
for k in range(2):
    for i in range(episodes):
        trainMonteCarlo((begin(),begin()))
    
    V = np.zeros((11, 22))
    for i in range(1,11):
        for j in range(1,22):
            V[i][j] = ((max_dict(Q[(i,j)])[1]))
    print("")
    print("Monte carlo V after ", episodes, " episodes")
    
    X = np.arange(1, 11)
    Y = np.arange(1, 22)
    X, Y = np.meshgrid(X, Y)
    Z = V[X,Y]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='winter_r')
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('Dealer')
    ax.set_ylabel('Player')
    ax.set_zlabel('V*')
    plt.show()
    episodes *= 10
    if k == 1:
        monteQ = copy.deepcopy(Q)
    Nstate,N,Q = init()
#------------------------
#   train sarsa and montecarlo
#------------------------   
total_sum = 0
meanSquare1000Ep = np.zeros(11)
lambdamat = {}
for k in range(0,11):
    lda = k/10
    lambdamat[k] = {}
    for episode in range(1,10001):
        trainSarsa((begin(),begin()))
        for s in state:
            for a in action:
                total_sum += np.power((Q[s][a]-monteQ[s][a]),2)
        lambdamat[k][episode]=(1/(len(action)*len(state))*total_sum)
        if episode == 10000:
            meanSquare1000Ep[k] = (lambdamat[k][episode])
        total_sum = 0
    Nstate,N,Q = init()

lambdamat[11] = {}
for episode in range(1,10001):
    trainMonteCarlo((begin(),begin()))
    for s in state:
        for a in action:
            total_sum += np.power((Q[s][a]-monteQ[s][a]),2)
    lambdamat[11][episode]=(1/(len(action)*len(state))*total_sum)
    total_sum = 0
    
#------------------------
#   Plot mean square vs episodes for sarsa all lambda and monte carlo
#------------------------
    
Nstate,N,Q = init()
fig = plt.figure(figsize=(12,12))
for i in range(10):
    plt.step(range(len(lambdamat[i])),lambdamat[i].values(),label='lambda ' + str(i/10), color = 'C' + str(i))
plt.step(range(len(lambdamat[10])),lambdamat[10].values(),label='lambda ' + str(1), color = 'k')
plt.step(range(len(lambdamat[11])),lambdamat[11].values(),label='Monte Carlo', color = 'saddlebrown')
plt.xlabel("Episodes")
plt.ylabel("Mean squared error")
plt.legend()
plt.show()
#------------------------
#   Plot mean square vs lambda 0,0.1,...,1 after 10000 episodes
#-----
plt.plot(np.arange(0.0,1.1,0.1),meanSquare1000Ep)
plt.xlabel("Lambda")
plt.ylabel("Mean squared error")
plt.ylim((0,0.5))
plt.show()

#------------------------
# Plot sarsa V for lambda 0, 0.5 and 1, 100000 episodes
#------------------------
for k in range(3):
    lda = k/2
    Nstate,N,Q = init()
    print("Sarsa V after 100000 episodes, Lambda = ", lda)
    for i in range(1000000):
        trainSarsa((begin(),begin()))
    
    V = np.zeros((11, 22))
    for i in range(1,11):
        for j in range(1,22):
            V[i][j] = ((max_dict(Q[(i,j)])[1]))
                
    X = np.arange(1, 11)
    Y = np.arange(1, 22)
    X, Y = np.meshgrid(X, Y)
    Z = V[X,Y]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='cool', linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('Dealer')
    ax.set_ylabel('Player')
    ax.set_zlabel('V')
    plt.show() 