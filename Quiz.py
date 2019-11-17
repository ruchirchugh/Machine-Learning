# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:04:24 2019

"""

import scipy.io as sio
datapath = 'tutorialRevLearn_low_s014_data.mat'
data = sio.loadmat(datapath, squeeze_me=True, struct_as_record=False)

import numpy as np
# choices = data.choice - 1 (black dots) 
# print(data.keys())
choices = data['data'].choice
modifiedchoices = choices - 1
r = data['data'].prep.feedback

# probability of reward of one of the machines 
probblue = data['data'].prep.feedbackprob
probblue

trials = list(range(1, len(modifiedchoices)+1, 1))
import matplotlib.pyplot as plt
plt.plot(trials, probblue, '-', trials, modifiedchoices, '+')
plt.show()

import statsmodels.api as sm
lowess = sm.nonparametric.lowess
smoothedchoice = lowess(modifiedchoices, trials)

plt.plot(trials, smoothedchoice[:, 1], 'r--', trials, probblue, '-', trials, modifiedchoices, '+')
plt.show()

# Want to make sure that exactly 1 machine gives a coin at every trial 
test = np.sum(r, axis = 1)
test
# we can see how many 1's there are in this case because the dataset is small enough 
# in practice, datasets can be very large, so to confirm that test indeed is an array of sums, we do the following:
if np.sum(test).astype(int) == len(test):
    print(True)
else:
    print(False)
    
# Bayesian Learning

H = 1/25 # as per the tutorial - do not know what a good way to decide this is

q = np.arange(0.01, 0.99, 0.01)
def Bayes(H):
    
# Let q be the probability that machine 0 gives a coin 
    q = np.arange(0.01, 0.99, 0.01)

# the prior distribution is uniform 
    prior = np.zeros((135, len(q)))
    prior[0] = (1/len(q))*np.ones(len(q))
    post = np.zeros((134, len(q)))
    
# p[i] is the prob distr for q for trial[i]  

    for i in range(0, 134):
        if r[i, 1] == 1: 
            post[i] = np.multiply(q, prior[i])/np.sum(np.multiply(q, prior[i])) # normalised posterior
        else:
            post[i] = np.multiply(1-q, prior[i])/np.sum(np.multiply(1-q, prior[i]))

# now for the prior for trial i + 1 
        prior[i+1] = (1-H)*post[i] + H*np.ones(len(q))/(len(q))
    return prior


# Plot prior distributions for each trial together with the expected value of q
# We expect these to be polynomials in q of increasing degree

H = 0.04

# Expected value of q using priors for trial i
eq = np.zeros(135)
for i in list(range(0, 135)): 
    eq[i] = np.sum(np.multiply(Bayes(H)[i], q))

eqplt = np.zeros((len(eq), len(q)))
for k in list(range(0, 135)):
    eqplt[k, :] = eq[k]*np.ones(len(q))
    
for i in range(0, 5):
    plt.plot(q, Bayes(H)[i], 'b-', eqplt[i], Bayes(H)[i], 'r--')
    plt.show()
    
probor = 1-probblue
plt.plot(trials, probor, 'b-', trials, eq, 'r--')
plt.show()

# There is a point of reversal ar around trial 100. We choose that. So we want the model to decrease its estimate of q.
a = list(range(99, 105, 1))
# plot prior probs against q tgt with the estimates of q
for i in range(99, 105):
    plt.plot(q, Bayes(H)[i], 'b-', eqplt[i], Bayes(H)[i], 'r--')
    plt.show()
    
