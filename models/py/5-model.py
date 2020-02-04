# Infers the two alpha's and a vector of C's.

# Uses an explicit formula for the utility term, instead of simulating inference about possible worlds
from math import log, exp
import torch
from pyro.infer import SVI
from pyro.optim import Adam
from torch.autograd import Variable

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

import prepareNgramData
from prepareNgramData import agreement

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI
from pyro.optim import Adam


n_adj = len(agreement)

n_obj = 20

n_speaker = 2

print agreement



import numpy.random

alpha1 = Variable(torch.FloatTensor([0.0]), requires_grad=True)
alpha2 = Variable(torch.FloatTensor([0.0]), requires_grad=True)

C_logit = Variable(torch.FloatTensor([-1.0]*n_adj), requires_grad=True)

optim = torch.optim.Adam([alpha1,alpha2, C_logit], lr=0.001)




## Distribution over listener / third-party beliefs about the object
## Given the adjectives A1, A2 and the object o given in the sentence, returns
## the joint distribution of A1(s2, o) and A2(s2, o).
## Here, while s1 is the speaker of the utterance, s2 is the belief of another speaker
## (or possibly of the listener, depending on the interpretation of the model.)
def posteriorRestrictionToObjectsAndAdjectivesForSent(sentence, loss2, adj1, adj2, C):
    kappa1 = agreement[sentence[0]]
    kappa2 = agreement[sentence[1]]
#    print ".."
#    print C
    C1 = C[adj1]
    C2 = C[adj2]
    p_A1_s0 = C1/(1-pow(C1,n_obj))
    #print p_A1_s0
    p_A2_s0 = 1.0

    p_A1_s1 = (1-kappa1) * C1 + kappa1 * p_A1_s0
    p_A2_s1 = (1-kappa2) * C2 + kappa2

    p_A1_s1_speaker = (1-kappa1) * C1 + kappa1
    p_A2_s1_speaker = (1-kappa2) * C2 + kappa2


#    print [p_A1_s0, p_A1_s1, p_A2_s0, p_A2_s1]
    eventsForS1 = [p_A1_s1 * p_A2_s1, (1-p_A1_s1) * p_A2_s1, p_A1_s1 * (1-p_A2_s1) ,(1-p_A1_s1) * (1-p_A2_s1)]
    surprisals = map(torch.log, eventsForS1)
    probabilities = [p_A1_s1_speaker * p_A2_s1_speaker, (1-p_A1_s1_speaker) * p_A2_s1_speaker, p_A1_s1_speaker * (1-p_A2_s1_speaker) ,(1-p_A1_s1_speaker) * (1-p_A2_s1_speaker)]
    return sum(map(lambda x : x[0]*x[1], zip(surprisals, probabilities)))

#    print "PROBABILITIES "+str(eventsForS1)
#    return sum(map(lambda x:x * log(x), eventsForS1))
 
#print posteriorRestrictionToObjectsAndAdjectivesForSent([0,1,1], True, 0, 1)
#print posteriorRestrictionToObjectsAndAdjectivesForSent([1,0,1], True, 1, 0)

def sigma(x):
   return 1/(1+exp(-x))

u = 0
c = 0
logAvg = -0.6931471805599453
correct = .5
for epoch in range(200):
  for point in prepareNgramData.dataset():
     C = nn.Sigmoid()(C_logit)
     adj1 = point["AdjNum1"]
     adj2 = point["AdjNum2"]
     pmi1 = point["PMI1"]
     pmi2 = point["PMI2"]
     utilityLoss12 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj1, adj2, 1], True, adj1, adj2,C)
     utilityLoss21 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj2, adj1, 1], True, adj2, adj1,C)
#     print [utilityLoss12, utilityLoss21, "AGR", agreement[adj1], agreement[adj2]]
     if (utilityLoss12.data.numpy()[0] > utilityLoss21.data.numpy()[0]) != (agreement[adj1] < agreement[adj2]):
         print ["ERROR", agreement[adj1], agreement[adj2]]
#     print [utilityLoss12 - utilityLoss21, agreement[adj1] - agreement[adj2]]
#     utilityLoss12 = agreement[adj1] - agreement[adj2]
#     utilityLoss21 = agreement[adj2] - agreement[adj1]

     costLoss12 = -pmi2
     costLoss21 = -pmi1
#     u += utilityLoss12 - utilityLoss21
#     c += costLoss12 - costLoss21
     logit = alpha1 * (utilityLoss12 - utilityLoss21) + alpha2 * (costLoss12 - costLoss21)
     log_probability = nn.LogSigmoid()(logit)
 #    print log_probability.data.numpy()[0]

     print "\t".join(map(str,[alpha1.data.numpy()[0], alpha2.data.numpy()[0], exp(-logAvg), correct]))
     print C.data.numpy()
     logAvg = 0.99 * logAvg + 0.01 * log_probability.data.numpy()[0]
     correct = 0.99 * correct + 0.01  * (1 if logit.data.numpy()[0] > 0 else 0)


     loss = -log_probability 
     optim.zero_grad()
     loss.backward()
     optim.step() 

 #    print [point["Adj1"],point["Adj2"],utilityLoss12, utilityLoss21, pmi1, pmi2, logit, probability]
#     d1Logit = (utilityLoss12 - utilityLoss21)
#     d2Logit = (costLoss12 - costLoss21)
#     outerDerivative = (1-exp(log_probability.data.numpy()[0]))
#     grad1 = outerDerivative * d1Logit
#     grad2 = outerDerivative * d2Logit
#     alpha1 += .001 * grad1
#     alpha2 += .001 * grad2


