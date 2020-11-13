# Using tabular representation for k=1 sparsity, og reward, low noise

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import math
import scipy.stats as ss
from scipy.stats import invgauss

class ActiveSearchRL(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.state_dim = 30
    self.b = Belief(n=self.state_dim)
    self.num_actions = 2
    self.observation_dim = len(self.b.sparse_vecs)
    self.verbose = True
    self.action_space = spaces.Box( np.array([0,0]), np.array([self.state_dim,self.state_dim]))
    self.observation_space = spaces.Box( np.array([0]*len(self.b.sparse_vecs)), np.array([1]*len(self.b.sparse_vecs)))
    self.timer = 0

  def reset(self):
      self.timer = 0
      self.b = Belief(n=self.state_dim)
      self.beta_star = self.b.sample()
      if self.verbose:
          print('\n')
          for j in self.beta_star:
              if j == 1:
                  print('*',end="")
              else:
                  print('.',end="")
          print('\n') 
      return np.asarray(self.b.representation()).flatten()

  def step(self,action):
          x = np.zeros(self.state_dim)

          self.timer += 1
          
          interval_start = int(abs(action[1]))
          interval_length = min( max(int(abs(action[0])),1), self.state_dim-interval_start)
          
          # if np.random.rand() < 0.05:
          #     interval_start = np.random.randint(self.state_dim)
          #     interval_length = min( max( np.random.randint(self.state_dim) ,1), self.state_dim-interval_start)  

          interval = np.ones(interval_length) / interval_length
          x[interval_start:interval_start+interval_length] = interval
          
          if self.verbose:
              print('\r',end="")
              for i in x:
                  if i == 0:
                      print('.', end="")
                  else:
                      print(1, end="")
              print('\r')
              
          y = (self.beta_star @ x) + (np.random.normal()*self.b.sigma)
          self.b.filt(x,y)
          reward = self.b.true_reward(self.beta_star)
          # bonus = interval_length*(300 - self.timer)/(200*self.state_dim)
          # for i in range(self.num_reward_samples):
          #     reward -= np.sqrt( np.mean( (self.b.sample() - self.beta_star)**2 ) )
          # reward /= self.num_reward_samples
          done = self.timer > 20
          if done:
              map_est = self.b.sparse_vecs[np.argmax(self.b.prior)]
              print('\n')
              print('-'*30)
              print('\r',end="")
              for i in map_est:
                  if i == 0:
                      print('.', end="")
                  else:
                      print('?', end="")
              print('\r')
              for i in self.beta_star:
                  if i == 0:
                      print('.', end="")
                  else:
                      print('*', end="")
              print('\r')
              print('-'*30)
          return np.asarray(self.b.representation()).flatten(), reward, done, {}

class Belief():
    def __init__(self,n=1,sigma=.00001,eta=2,num_representation_draws=100, lmbd=1, itr=100):
        self.X = []
        self.y = []
        self.n = n
        self.sigma = sigma
        self.eta = eta
        self.num_representation_draws = num_representation_draws
        self.lmbd = lmbd
        self.itr = itr

        sparse_vecs = set()

        # for i in range(self.n):
        #     for j in range(self.n):
        #         sparse_vec = np.zeros(self.n)
        #         sparse_vec[i] = 1
        #         sparse_vec[j] = 1
        #         sparse_vecs.add(tuple(sparse_vec))

        for i in range(self.n):
            sparse_vec = np.zeros(self.n)
            sparse_vec[i] = 1
            sparse_vecs.add(tuple(sparse_vec))

        self.sparse_vecs = np.asarray(list(sparse_vecs))
        self.prior = np.ones(len(self.sparse_vecs))/len(self.sparse_vecs)
    
    def filt(self,x,y):
        self.X.append(x)
        self.y.append(y)
        prior = self.prior
        posterior = np.zeros(len(self.sparse_vecs))
        for j in range(len(posterior)):
            posterior[j] =  self.gauss_pdf( y - self.sparse_vecs[j].T @ x )*prior[j]
        posterior = posterior/np.sum(posterior)
        self.prior = posterior

    def gauss_pdf(self,x):
        return np.exp(-0.5*((x/self.sigma)**2))
    
    # def sample(self):
    #     tau = np.random.exponential(2/self.eta,size=self.n)
    #     X = np.asarray(self.X)
    #     y = np.asarray(self.y).flatten()
    #     if len(self.X) == 0:
    #         return( np.random.multivariate_normal(np.zeros(self.n),np.diag(tau)) )
    #     Sig = np.linalg.inv( np.linalg.inv(np.diag(tau)) + (1/(self.sigma)**2) * X.T @ X )
    #     mu = (1/(self.sigma)**2) * Sig @ X.T @ y
    #     return( np.random.multivariate_normal(mu,Sig) )


    # def sample(self):
    #     X = np.asarray(self.X)
    #     y = np.asarray(self.y).flatten()        
        
    #     if len(y) == 0:
    #         beta_tilde = np.random.laplace(scale=1/self.lmbd,size=self.n)
    #         return beta_tilde
        
    #     XT_X = X.T @ X
    #     XT_Y = X.T @ y
        
    #     tauinv_vec = 1/np.random.rand(self.n)
    #     score = 0
    #     for i in range(self.itr):
    #         Sig = np.linalg.inv(XT_X+(self.sigma**2)*np.diag(tauinv_vec)+1e-3*np.eye(self.n))
    #         beta = np.random.multivariate_normal(np.squeeze(np.matmul(Sig,XT_Y)),(self.sigma**2)*Sig)
    #         for j in range(self.n):
    #             tauinv_vec[j] = invgauss.rvs(np.sqrt((self.sigma**2))*(self.lmbd**(1/3))/np.abs(beta[j]))*(self.lmbd**(2/3))
    #     return beta 
    def true_reward(self,beta):
        expectation = 0
        for index, sparse_vec in enumerate(self.sparse_vecs):
            expectation += self.prior[index] * np.mean((sparse_vec - beta)**2)
        return -1*expectation
        # entropy = 0
        # for index, sparse_vec in enumerate(self.sparse_vecs):
        #     if self.prior[index] > 0:
        #         entropy -= self.prior[index] * np.log(self.prior[index])

        # return -1*entropy


    def sample(self):
        return self.sparse_vecs[np.random.choice(np.arange(len(self.sparse_vecs)), p=self.prior )]          
    
    def representation(self):
        return self.prior

        # samples = []
        # for i in range(self.num_representation_draws):
        #     samples.append(self.sample())
        # # representation is estimate of the mean and diagonlized covariance of the posterior
        # return np.mean(samples,axis=0) , np.diag(np.cov(np.asarray(samples).T)) 