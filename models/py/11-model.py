# Predictions from 8

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




import numpy.random



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
    p_A1_s0 = C[adj1]/(1-pow(1-C[adj1],n_obj))
    #print p_A1_s0
    p_A2_s0 = 1.0

    p_A1_s1 = (1-kappa1) * C[adj1] + kappa1 * p_A1_s0
    p_A2_s1 = (1-kappa2) * C[adj2] + kappa2

    p_A1_s1_speaker = (1-kappa1) * C[adj1] + kappa1
    p_A2_s1_speaker = (1-kappa2) * C[adj2] + kappa2


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
    mu1 = pyro.param("alpha1_mu",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu2 = pyro.param("alpha2_mu",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu3 = pyro.param("C_logit_mu",Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu4 = pyro.param("C_per_adj_var_mu", Variable(torch.Tensor([(0.0)]), requires_grad=True))
    mu5 = pyro.param("C_per_adj_mu", Variable(torch.Tensor([0.0]*n_adj), requires_grad=True))

  
    sigma1 = pyro.param("alpha1_sigma", Variable(torch.Tensor([(3.0)]), requires_grad=True))
    sigma2 = pyro.param("alpha2_sigma", Variable(torch.Tensor([(3.0)]), requires_grad=True))
    sigma3 = pyro.param("C_logit_sigma", Variable(torch.Tensor([(0.5)]), requires_grad=True))
    sigma4 = pyro.param("C_per_adj_var_sigma", Variable(torch.Tensor([(0.5)]), requires_grad=True))
    sigma5 = pyro.param("C_per_adj_sigma", Variable(torch.Tensor([(0.5)]*n_adj), requires_grad=True))

#    alpha_C_1_log = pyro.param("alpha_C_1_log", Variable(torch.Tensor([(1.0)]), requires_grad=True))
#    beta_C_1_log = pyro.param("beta_C_1_log", Variable(torch.Tensor([(1.0)]), requires_grad=True))


    pyro.sample("alpha1", dist.normal, mu1, sigma1)
    pyro.sample("alpha2", dist.normal, mu2, sigma2)
#    pyro.sample("C_logit", dist.beta, torch.exp(alpha_C_1_log), torch.exp(beta_C_1_log))
    pyro.sample("C_logit", dist.normal, mu3, sigma3)
    pyro.sample("C_per_adj_var", dist.normal, mu4, sigma4)
    pyro.sample("C_per_adj", dist.normal, mu5, sigma5)


mu_alpha1, sigma_alpha1 = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([20.0]))
mu_alpha2, sigma_alpha2 = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([20.0]))

alpha1_prior = Normal(mu_alpha1, sigma_alpha1)
alpha2_prior = Normal(mu_alpha2, sigma_alpha2)

# prior for the mean
mu_C_logit, sigma_C_logit = Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([1.0]))

# prior for the variance of the per-adjective Cs
C_random_variance_prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([1.0])))

C_logit_prior = Normal(mu_C_logit, sigma_C_logit)

# parameters from 8: ~/scr/qp/pyro-8-model-out-1.txt

print "\t".join(map(str,["Adj1", "Adj2", "UtilityDiff", "CostDiff", "Logit", "SubjDiff", "C_Adj1", "C_Adj2"]))
#point["Adj1"], point["Adj2"], (utilityLoss12-utilityLoss21).data.numpy()[0], (costLoss12-costLoss21), logit.data.numpy()[0], (agreement[adj1] - agreement[adj2]),  C[adj1].data.numpy()[0], C[adj2].data.numpy()[0]]))


def model(dataset):
  #alpha1 = Variable(torch.FloatTensor([0.0]), requires_grad=True)
  alpha1 = Variable(torch.FloatTensor([ 5.60013199]))
  #alpha2 = Variable(torch.FloatTensor([0.0]), requires_grad=True)
  alpha2 = Variable(torch.FloatTensor([-0.36089259]))

  # the intercept
  C_logit = Variable(torch.FloatTensor([-2.09661484]))

  C_offsets_per_adj_variance = pyro.sample("C_per_adj_var", C_random_variance_prior)

  C_offsets_per_adj = Variable(torch.FloatTensor([ -5.69128171e-02,  -1.82613637e-02,  -1.21957331e-03,
         1.17304558e-02,  -2.48900373e-02,  -1.10827647e-02,
        -3.46694016e+00,   6.87145023e-03,   2.60550880e+00,
        -4.39196564e-02,  -4.42069829e-01,  -1.49831367e+00,
         2.60351062e-01,  -7.75251314e-02,   6.88868156e-03,
        -1.04497541e-02,  -3.17913406e-02,   1.86948478e-02,
         1.34600671e-02,   2.60929111e-02,  -2.06632651e-02,
        -1.83618721e-02,   8.80209625e-01,   5.50859608e-02,
         1.42775616e-02,   2.44544938e-01,   3.20089161e-02,
         2.96289660e-02,  -8.83045781e-04,   1.91952381e-03,
         4.35157912e-03,   6.67811465e-03,   9.34787512e-01,
        -2.96084266e-02,   2.06944227e+00,  -4.11597490e-01,
        -2.44996834e+00,  -9.59226768e-03,   3.55667561e-01,
         3.20782274e-01,  -1.05066426e-01,   8.99812952e-03,
        -6.06162190e-01,   1.43420137e-02,   1.26842833e+00,
        -1.07140094e-02,   6.86585065e-03,   2.50395030e-01,
        -4.84525878e-03,  -3.04391440e-02,  -2.99848616e-02,
        -1.24874008e+00,  -3.96026611e-01,   4.36177909e-01,
        -1.97266793e+00,  -1.97245404e-02,   6.18222589e-03,
        -3.93819772e-02,   9.19246152e-02,  -1.60655305e-02,
         4.52995393e-03,  -6.10954463e-02,   8.32783163e-01,
        -1.88960426e-03,  -1.87259614e-01,  -7.04466760e-01,
        -5.85746067e-03,   2.22841371e-02,  -1.39677115e-02,
        -2.03311950e-01,   2.65220478e-02,   5.24445213e-02,
        -1.34638846e-01,  -1.67852545e+00,   9.79648650e-01,
        -1.43779129e-01,   2.79521257e-01,   3.33210230e-01,
         1.83970146e-02,  -2.32099630e-02,   6.14077896e-02,
         1.01985025e+00,  -6.71154866e-03,   2.68969327e-01,
         6.65332228e-02,  -8.95076334e-01,   9.97939289e-01,
         1.81760974e-02,   1.18259326e-01,  -3.08577031e-01,
        -1.47507846e+00,   1.39819637e-01,  -2.17066169e+00]))
  C = nn.Sigmoid()(C_logit + C_offsets_per_adj)

  for i in pyro.irange("data_loop", len(dataset)): #point in dataset:
     point = dataset[i]


     adj1 = point["AdjNum1"]
     adj2 = point["AdjNum2"]
     pmi1 = point["PMI1"]
     pmi2 = point["PMI2"]
     utilityLoss12 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj1, adj2, 1], True, adj1, adj2,C)
     utilityLoss21 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj2, adj1, 1], True, adj2, adj1,C)
#     print [utilityLoss12, utilityLoss21, "AGR", agreement[adj1], agreement[adj2]]
     if False:
       if (utilityLoss12.data.numpy()[0] > utilityLoss21.data.numpy()[0]) != (agreement[adj1] < agreement[adj2]):
         print ["ERROR", agreement[adj1], agreement[adj2]]

# python 11-model.py > ~/scr/qp/pyro-11-model-out-1.tsv

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
     print "\t".join(map(str,[point["Adj1"], point["Adj2"], (utilityLoss12-utilityLoss21).data.numpy()[0], (costLoss12-costLoss21), logit.data.numpy()[0], (agreement[adj1] - agreement[adj2]),  C[adj1].data.numpy()[0], C[adj2].data.numpy()[0]]))
#     pyro.observe("result_{}".format(i), Bernoulli(logits=logit), Variable(torch.FloatTensor([1.0])))
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


model(dataset)

#adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
#optimizer = Adam(adam_params)
#
## setup the inference algorithm
#svi = SVI(model, guide, optimizer, loss="ELBO") #, num_particles=7)
#
#n_steps = 40000
## do gradient steps
#for step in range(n_steps):
#    svi.step(dataset)
#    if step % 100 == 0:
#        print "......."
#        print step
#
#        for name in pyro.get_param_store().get_all_param_names():
#           print [name, pyro.param(name).data.numpy()]

#model(dataset)

quit()

#
## The marginal posterior probability of the judgment Adjective(Person, Object), after hearing an utterance
#def marginal1Enum = cache( lambda utterance, person, object, adjective) { Infer({method: 'enumerate',
#      model() {
#         def world = sample(third(utterance))
#         return world[adjective][object][person] ? 1 : 0
#      }})})
#
#
######################################
## Inspect the posterior by looking at the coordinate-wise marginals
######################################
#
## Recursively create the table of the posterior marginals
#def computeMarginalPerson(utterance, adj, obj, person) {
#   distribution = marginal1Enum(utterance, person, obj, adj).getDist()
#   if distribution['0'] != undefined  and  distribution['0']['val'] == 1:
#      return distribution['0']['prob']
#   else:
#      return distribution['1']['prob']
#   return result
#}
#
#def computeMarginalObj(utterance, adj, obj) {
#  if obj == n_obj:
#    return []
#  else:
#    first = map(lambda person : computeMarginalPerson(utterance, adj, obj, person), range(n_speaker))
#    result = [first] + computeMarginalObj(utterance, adj,obj+1)
#    return result
#
#def prettyPrint(utterance) {
#   adj1 = "Adjective"+utterance[0]
#   adj2 = "Adjective"+utterance[1]
#   noun = "Noun"+utterance[2]
#   return adj1+" "+adj2+" "+noun
#}
#
#
#
## Distribution over listener / third-party beliefs about the object
## Given the adjectives A1, A2 and the object o given in the sentence, returns
## the joint distribution of A1(s2, o) and A2(s2, o).
## Here, while s1 is the speaker of the utterance, s2 is the belief of another speaker
## (or possibly of the listener, depending on the interpretation of the model.)
#def posteriorRestrictionToObjectsAndAdjectivesForSent = cache( lambda sentence, lossProb2, lossProb1, adj1, adj2) {
#     Infer({method: 'enumerate', #'rejection', samples : 10, incremental:true,
#            model() {
#               def model = sample(third(sentence, lossProb2, lossProb1))
#               return [model[adj1][1][1], model[adj2][1][1]]
#           }})})
#
##console.log(restrictionToObjectsAndAdjectivesForSent(options[0]))
##console.log(restrictionToObjectsAndAdjectivesForSent(options[1]))
#
#
#
#def options(adj1, adj2) { return [[adj1, adj2, 1], [adj2, adj1, 1]] }
#
#def speaker(rationality, lossProb2, lossProb1, adj1, adj2):
#                   def speakerModel = Infer({method : 'enumerate',
#                       model() {
#                       def sentence = options(adj1, adj2)[sample(RandomInteger({n : 2}))]
#                       #console.log(posteriorRestrictionToObjectsAndAdjectivesForSent(sentence, lossProb2, lossProb1))
#                       factor(-rationality*entropy(posteriorRestrictionToObjectsAndAdjectivesForSent(sentence, lossProb2, lossProb1, adj1, adj2)))
#                       return sentence
#                    }})
#                    return speakerModel.getDist()['['+adj1+','+adj2+',1]'].prob
#
##   console.log("\nSpeaker Distribution: Subjective adjective is preferred earlier.")
##   console.log(prettyPrint([0,1,1])+"   "+speaker.getDist()['[0,1,1]'].prob)
##   console.log(prettyPrint([1,0,1])+"   "+speaker.getDist()['[1,0,1]'].prob)
#
#def applyToAdjectivePair(rationality, lossProb1, lossProb2, dataPair):
#     print "\t".join(map(str,[agreement[dataPair[0][0]], agreement[dataPair[0][1]], agreement[dataPair[0][0]]- agreement[dataPair[0][1]], speaker(rationality, lossProb2, lossProb1, dataPair[0][0], dataPair[0][1]), dataPair[1]]))
#


dataPairs = [
    [[11, 89], 0.3825],
    [[43, 84], 0.32],
    [[71, 51], 0.18],
    [[73, 89], 0.515],
    [[11, 52], 0.305],
    [[37, 68], 0.733333333333333],
    [[53, 84], 0.9225],
    [[11, 93], 0.145],
    [[11, 45], 0.468333333333333],
    [[67, 82], 0.388],
    [[13, 41], 0.508],
    [[81, 91], 0.86],
    [[10, 71], 0.215],
    [[41, 62], 0.718333333333333],
    [[14, 91], 0.37],
    [[54, 90], 0.194285714285714],
    [[40, 53], 0.363333333333333],
    [[54, 89], 0.4325],
    [[55, 21], 0.7425],
    [[73, 82], 0.834285714285714],
    [[1, 92], 0.43],
    [[92, 7], 0.63],
    [[63, 41], 0.84],
    [[41, 57], 0.852],
    [[60, 93], 0.5075],
    [[11, 80], 0.396],
    [[6, 81], 0.185454545454545]
]

rationality = 3
lossProb2 = 1.0
lossProb1 = 0.0
print [lossProb2, rationality]
#map( lambda pair) { return applyToAdjectivePair(rationality, lossProb1, lossProb2, pair) }, dataPairs   )





w = Variable(torch.zeros(1), requires_grad=True)

def model(x):
    pass

def guide(x):
    pass

opt = Adam({"lr": 0.0001})
loss = SVI(model, guide, opt, loss="ELBO")
loss.step(1)





for i in range(10):
   mu = Variable(torch.zeros(1))   # mean zero
   sigma = Variable(torch.ones(1)) # unit variance
   x = dist.normal(mu, sigma)      # x is a sample from N(0,1)
   print(x)



   log_p_x = dist.normal.log_pdf(x, mu, sigma)
   print(log_p_x)

   x = pyro.sample("my_sample", dist.normal, mu, sigma)
   print(x)


def weather():
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.3])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    mean_temp = {'cloudy': [55.0], 'sunny': [75.0]}[cloudy]
    sigma_temp = {'cloudy': [10.0], 'sunny': [15.0]}[cloudy]
    temp = pyro.sample('temp', dist.normal,
                       Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]

for _ in range(3):
    print(weather())

def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = [200] if cloudy == 'sunny' and temp > 80.0 else [50]
    ice_cream = pyro.sample('ice_cream', dist.normal,
                            Variable(torch.Tensor(expected_sales)),
                            Variable(torch.Tensor([10.0])))
    return ice_cream


def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), dist.bernoulli, p)
    if torch.equal(x.data, torch.zeros(1)):
        return x
    else:
        return x + geometric(p, t+1)

print(geometric(Variable(torch.Tensor([0.5]))))

def normal_product(mu, sigma):
    z1 = pyro.sample("z1", dist.normal, mu, sigma)
    z2 = pyro.sample("z2", dist.normal, mu, sigma)
    y = z1 * z2
    return y

def make_normal_normal():
    mu_latent = pyro.sample("mu_latent", dist.normal,
                            Variable(torch.zeros(1)),
                            Variable(torch.ones(1)))
    fn = lambda sigma: normal_product(mu_latent, sigma)
    return fn

print(make_normal_normal()(Variable(torch.ones(1))))


def scale(guess):
    # The prior over weight encodes our uncertainty about our guess
    weight = pyro.sample("weight", dist.normal, guess, Variable(torch.ones(1)))
    # This encodes our belief about the noisiness of the scale:
    # the measurement fluctuates around the true weight
    return pyro.sample("measurement", dist.normal, weight, Variable(torch.Tensor([0.75])))

posterior = pyro.infer.Importance(scale, num_samples=100)



guess = Variable(torch.Tensor([8.5]))

marginal = pyro.infer.Marginal(posterior)
print(marginal(guess))

import matplotlib.pyplot as plt

plt.hist([marginal(guess).data[0] for _ in range(100)], range=(5.0, 12.0))
plt.title("P(measurement | guess)")
plt.xlabel("weight")
plt.ylabel("#")
plt.show()












