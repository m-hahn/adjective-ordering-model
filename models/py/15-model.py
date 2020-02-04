# Creates predictions for ratings data, assuming fixed C=.2

# Uses an explicit formula for the utility term, instead of simulating inference about possible worlds
# Check whether formula is correct
from math import log, exp
import torch
from pyro.infer import SVI
from pyro.optim import Adam
from torch.autograd import Variable

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

import prepareScontrasData
from prepareSubjectivityData import agreement, adjToNum
import prepareMIData

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




import numpy.random




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
    p_A1_s0 = C/(1-pow(1-C,n_obj))
    #print p_A1_s0
    p_A2_s0 = 1.0


    # It is important that the KL divergence is computed separately for the posterior Marginals for the different speakers.
    # Otherwise, the KL divergence is just the (purely semantic) surprisal of the part that didn't get transmitted.
    p_A1_s1 = ((1-kappa1)*C + kappa1) * p_A1_s0
    p_not_A1_s1 = (1-((1-kappa1)*C + kappa1)) * p_A1_s0

    p_A2_s1 = (1-kappa2) * C + kappa2

    p_A1_s1_speaker = (1-kappa1)*C + kappa1
    p_A2_s1_speaker = (1-kappa2) * C + kappa2


#    print [p_A1_s0, p_A1_s1, p_A2_s0, p_A2_s1]
    eventsForS1 = [p_A1_s1 * p_A2_s1, p_not_A1_s1 * p_A2_s1, p_A1_s1 * (1-p_A2_s1) ,p_not_A1_s1 * (1-p_A2_s1)]
    surprisals = map(torch.log, eventsForS1)
    probabilities = [p_A1_s1_speaker * p_A2_s1_speaker, (1-p_A1_s1_speaker) * p_A2_s1_speaker, p_A1_s1_speaker * (1-p_A2_s1_speaker) ,(1-p_A1_s1_speaker) * (1-p_A2_s1_speaker)]
    surprisalsSpeaker = map(torch.log, probabilities)
    kl = sum(map(lambda x : (x[0][0]-x[0][1])*x[1], zip(zip(surprisals, surprisalsSpeaker), probabilities)))
    print kl
    return kl

#    print "PROBABILITIES "+str(eventsForS1)
#    return sum(map(lambda x:x * log(x), eventsForS1))
 
#print posteriorRestrictionToObjectsAndAdjectivesForSent([0,1,1], True, 0, 1)
#print posteriorRestrictionToObjectsAndAdjectivesForSent([1,0,1], True, 1, 0)

def sigma(x):
   return 1/(1+exp(-x))


#dataset_size = sum(1 for _ in prepareNgramData.dataset())
dataiter = prepareScontrasData.dataset()
dataset = [x for x in dataiter]

u = 0
c = 0
logAvg = -0.6931471805599453
correct = .5
#for epoch in range(200):

def guide(dataset):
    # register the two variational parameters with Pyro
    mu1 = pyro.param("alpha1_mu",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu2 = pyro.param("alpha2_mu",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu3 = pyro.param("C_logit_mu",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu4 = pyro.param("intercept_mu",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu5 = pyro.param("log_variance_mu",Variable(torch.Tensor([(-1.0)]), requires_grad=True))

    sigma1 = pyro.param("alpha1_sigma", Variable(torch.Tensor([(3.0)]), requires_grad=True))
    sigma2 = pyro.param("alpha2_sigma", Variable(torch.Tensor([(3.0)]), requires_grad=True))
    sigma3 = pyro.param("C_logit_sigma", Variable(torch.Tensor([(0.5)]), requires_grad=True))
    sigma4 = pyro.param("intercept_sigma", Variable(torch.Tensor([(0.5)]), requires_grad=True))
    sigma5 = pyro.param("log_variance_sigma", Variable(torch.Tensor([(0.5)]), requires_grad=True))

#    alpha_C_1_log = pyro.param("alpha_C_1_log", Variable(torch.Tensor([(1.0)]), requires_grad=True))
#    beta_C_1_log = pyro.param("beta_C_1_log", Variable(torch.Tensor([(1.0)]), requires_grad=True))


    pyro.sample("alpha1", dist.normal, mu1, sigma1)
    pyro.sample("alpha2", dist.normal, mu2, sigma2)
#    pyro.sample("C_logit", dist.beta, torch.exp(alpha_C_1_log), torch.exp(beta_C_1_log))
    pyro.sample("C_logit", dist.normal, mu3, sigma3)
    pyro.sample("intercept", dist.normal, mu4, sigma4)
    pyro.sample("log_variance", dist.normal, mu5, sigma5)


mu_alpha1, sigma_alpha1 = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([20.0]))
mu_alpha2, sigma_alpha2 = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([20.0]))

alpha1_prior = Normal(mu_alpha1, sigma_alpha1)
alpha2_prior = Normal(mu_alpha2, sigma_alpha2)

mu_C_logit, sigma_C_logit = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([1.0]))
C_logit_prior = Normal(mu_C_logit, sigma_C_logit)

intercept_prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([1.0])))
variance_prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([1.0])))



print "\t".join(map(str,["Worker","Adj1","Adj2","PMI_Diff", "Utility_Diff", "Rating"]))

def model(dataset):
  #alpha1 = Variable(torch.FloatTensor([0.0]), requires_grad=True)
  alpha1 = pyro.sample("alpha1", alpha1_prior)
  #alpha2 = Variable(torch.FloatTensor([0.0]), requires_grad=True)
  alpha2 = pyro.sample("alpha2", alpha2_prior)

  intercept = pyro.sample("intercept", intercept_prior)
  log_variance = pyro.sample("log_variance", variance_prior)

  C_logit = pyro.sample("C_logit", C_logit_prior) #nn.Sigmoid()(C_logit)
  C = Variable(torch.FloatTensor([0.1])) #nn.Sigmoid()(C_logit)

  for i in pyro.irange("data_loop", len(dataset)): #point in dataset:
     point = dataset[i]

     adj1Tok = point["Adj1"]
     adj2Tok = point["Adj2"]

     adj1 = adjToNum[adj1Tok] #point["AdjNum1"]
     adj2 = adjToNum[adj2Tok] #point["AdjNum2"]
#     print [adj1Tok, agreement[adj1]]
#     print [adj2Tok, agreement[adj2]]

     pmi1 = prepareMIData.getPMIUpToConstant(adj1Tok, point["Noun"]) #point["PMI1"]
     pmi2 = prepareMIData.getPMIUpToConstant(adj2Tok, point["Noun"])
     utilityLoss12 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj1, adj2, 1], True, adj1, adj2,C)
     utilityLoss21 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj2, adj1, 1], True, adj2, adj1,C)
#     print [utilityLoss12, utilityLoss21, "AGR", agreement[adj1], agreement[adj2]]
     if (utilityLoss12.data.numpy()[0] > utilityLoss21.data.numpy()[0]) != (agreement[adj1] < agreement[adj2]):
         print ["ERROR", agreement[adj1], agreement[adj2]]


     costLoss12 = -pmi2
     costLoss21 = -pmi1

     logit = intercept + alpha1 * (utilityLoss12 - utilityLoss21) + alpha2 * (costLoss12 - costLoss21)
     print "\t".join(map(str,[point["worker"],adj1Tok,adj2Tok,pmi1-pmi2,utilityLoss12.data.numpy()[0] - utilityLoss21.data.numpy()[0], point["rating"], agreement[adj1], agreement[adj2]]))
     


     log_probability = nn.LogSigmoid()(logit)
     #probability = nn.Sigmoid()(logit)
 #    result = Bernoulli(logits=logit)
#     print pyro.sample("resu", result)
#     print "LOGIT"
#     print logit
#     result = pyro.sample("result_"+str(i),  Bernoulli(logits=logit))
#     pyro.observe("result_{}".format(i), Normal( logit, torch.exp(log_variance)),  Variable(torch.FloatTensor([point["rating"]]) ))
     #print point["rating"]

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

model(dataset)



