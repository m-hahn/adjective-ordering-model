# Infers the two alpha's, with fixed C=0.2.
# Variational inference in Pyro.

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
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI
from pyro.optim import Adam


n_adj = len(agreement)

n_obj = 20

n_speaker = 2

print agreement



import numpy.random



mu_C_logit, sigma_C_logit = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([1.0]))

C_logit_prior = Normal(mu_C_logit, sigma_C_logit)
#priors = 

#C_logit = Variable(torch.FloatTensor([0.0]), requires_grad=True)

#optim = torch.optim.Adam([alpha1,alpha2, C_logit], lr=0.001)
#optim = torch.optim.Adam([C_logit], lr=0.0000000000001)


#class SpeakerModel(nn.Module):
#   def __init__(self):
#        super(SpeakerModel, self).__init__()
#        self.alpha1 = alpha




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
    p_A1_s0 = C/(1-pow(C,n_obj))
    #print p_A1_s0
    p_A2_s0 = 1.0

    p_A1_s1 = (1-kappa1)*C + kappa1 * p_A1_s0
    p_A2_s1 = (1-kappa2) * C + kappa2

    p_A1_s1_speaker = (1-kappa1)*C + kappa1
    p_A2_s1_speaker = (1-kappa2) * C + kappa2


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


#dataset_size = sum(1 for _ in prepareNgramData.dataset())
dataiter = prepareNgramData.dataset()
dataset = [x for x in dataiter]

u = 0
c = 0
logAvg = -0.6931471805599453
correct = .5
#for epoch in range(200):

def guide(dataset):
    # register the two variational parameters with Pyro
    mu1 = pyro.param("mu1",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu2 = pyro.param("mu2",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    sigma1 = pyro.param("sigma1", Variable(torch.Tensor([(3.0)]), requires_grad=True))
    sigma2 = pyro.param("sigma2", Variable(torch.Tensor([(3.0)]), requires_grad=True))

    pyro.sample("alpha1", dist.normal, mu1, sigma1)
    pyro.sample("alpha2", dist.normal, mu2, sigma2)

mu_alpha1, sigma_alpha1 = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([20.0]))
mu_alpha2, sigma_alpha2 = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([20.0]))

alpha1_prior = Normal(mu_alpha1, sigma_alpha1)
alpha2_prior = Normal(mu_alpha2, sigma_alpha2)


C = Variable(torch.FloatTensor([0.2])) #nn.Sigmoid()(C_logit)

def model(dataset):
  #alpha1 = Variable(torch.FloatTensor([0.0]), requires_grad=True)
  alpha1 = pyro.sample("alpha1", alpha1_prior)
  #alpha2 = Variable(torch.FloatTensor([0.0]), requires_grad=True)
  alpha2 = pyro.sample("alpha2", alpha2_prior)

  for i in pyro.irange("data_loop", len(dataset), subsample_size = 5): #point in dataset:
     point = dataset[i]


     adj1 = point["AdjNum1"]
     adj2 = point["AdjNum2"]
     pmi1 = point["PMI1"]
     pmi2 = point["PMI2"]
     utilityLoss12 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj1, adj2, 1], True, adj1, adj2,C)
     utilityLoss21 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj2, adj1, 1], True, adj2, adj1,C)
#     print [utilityLoss12, utilityLoss21, "AGR", agreement[adj1], agreement[adj2]]
     if (utilityLoss12.data.numpy()[0] > utilityLoss21.data.numpy()[0]) != (agreement[adj1] < agreement[adj2]):
         print ["ERROR", agreement[adj1], agreement[adj2]]


     costLoss12 = -pmi2
     costLoss21 = -pmi1

     logit = alpha1 * (utilityLoss12 - utilityLoss21) + alpha2 * (costLoss12 - costLoss21)
     log_probability = nn.LogSigmoid()(logit)
     #probability = nn.Sigmoid()(logit)
 #    result = Bernoulli(logits=logit)
#     print pyro.sample("resu", result)
#     print "LOGIT"
#     print logit
#     result = pyro.sample("result_"+str(i),  Bernoulli(logits=logit))
     pyro.observe("result_{}".format(i), Bernoulli(logits=logit), Variable(torch.FloatTensor([1.0])))
 #    print result    

 #    pyro.condition(result, data={"result_"+str(i) : Variable(torch.FloatTensor([1.0]))})

 #    print log_probability.data.numpy()[0]
     global logAvg
     global correct
#     print "\t".join(map(str,[C.data.numpy()[0], alpha1.data.numpy()[0], alpha2.data.numpy()[0], exp(-logAvg), correct]))
     logAvg = 0.99 * logAvg + 0.01 * log_probability.data.numpy()[0]
     correct = 0.99 * correct + 0.01  * (1 if logit.data.numpy()[0] > 0 else 0)


     loss = -log_probability 
#     optim.zero_grad()
#     loss.backward()
#     optim.step() 



adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss="ELBO") #, num_particles=7)

n_steps = 40000
# do gradient steps
for step in range(n_steps):
    svi.step(dataset)
    if step % 100 == 0:
        print "......."
        print step

        for name in pyro.get_param_store().get_all_param_names():
           print("[%s]: %.3f" % (name, pyro.param(name).data.numpy()))


