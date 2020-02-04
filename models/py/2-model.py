# Could also take only four worlds, do something more sophisticated when computing the posterior, and reweighting worlds based on C. This way, one could do inference over C.


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


n_adj = len(agreement)

n_obj = 20

n_speaker = 2

print agreement

#agreement = [0.1, 0.9, 0.347666666666667, 0.48479591836734703, 0.384615384615385, 0.44899999999999995, 0.561875, 0.430921052631579, 0.5676923076923079, 0.369661016949153, 0.795729166666667, 0.813666666666667, 0.7785, 0.801153846153846, 0.761894736842105, 0.7831372549019611, 0.781216216216216, 0.7044736842105259, 0.750540540540541, 0.815333333333333, 0.863666666666667, 0.780740740740741, 0.811358024691358, 0.781052631578947, 0.795070422535211, 0.24066666666666703, 0.211666666666667, 0.771, 0.820442477876106, 0.768260869565217, 0.318333333333333, 0.320666666666667, 0.30200000000000005, 0.319666666666667, 0.354, 0.34533333333333305, 0.40166666666666695, 0.456666666666667, 0.400333333333333, 0.404761904761905, 0.44764705882352895, 0.41057142857142903, 0.46558823529411797, 0.5133802816901409, 0.628333333333333, 0.437536231884058, 0.372714285714286, 0.2875, 0.216233766233766, 0.5401369863013701, 0.29259259259259296, 0.405, 0.374307692307692, 0.296103896103896, 0.28014925373134303, 0.633108108108108, 0.6779999999999999, 0.696710526315789, 0.774133333333333, 0.784390243902439, 0.772205882352941, 0.741805555555556, 0.7405128205128211, 0.84958904109589, 0.695694444444444, 0.811756756756757, 0.5129729729729731, 0.494320987654321, 0.588767123287671, 0.79609756097561, 0.665625, 0.722033898305085, 0.284303797468354, 0.355125, 0.37279411764705905, 0.404320987654321, 0.35789473684210504, 0.540945945945946, 0.858805970149254, 0.628243243243243, 0.628360655737705, 0.20268292682926803, 0.226436781609195, 0.20144736842105304, 0.30303571428571396, 0.30851851851851897, 0.284153846153846, 0.327027027027027, 0.21830769230769198, 0.69375, 0.33571428571428596, 0.56, 0.7133333333333329, 0.5007352941176471, 0.32783783783783804, 0.566363636363636, 0.262272727272727]

CConst = 0.2


import numpy.random

## Distribution over listener / third-party beliefs about the object
## Given the adjectives A1, A2 and the object o given in the sentence, returns
## the joint distribution of A1(s2, o) and A2(s2, o).
## Here, while s1 is the speaker of the utterance, s2 is the belief of another speaker
## (or possibly of the listener, depending on the interpretation of the model.)
def posteriorRestrictionToObjectsAndAdjectivesForSent(sentence, loss2, adj1, adj2):
    kappa1 = agreement[sentence[0]]
    kappa2 = agreement[sentence[1]]
    p_A1_s0 = CConst/(1-pow(CConst,n_obj))
    p_A2_s0 = 1.0

    p_A1_s1 = (1-kappa1)*CConst + kappa1 * p_A1_s0
    p_A2_s1 = (1-kappa2) * CConst + kappa2

    p_A1_s1_speaker = (1-kappa1)*CConst + kappa1
    p_A2_s1_speaker = (1-kappa2) * CConst + kappa2

    

#    print [p_A1_s0, p_A1_s1, p_A2_s0, p_A2_s1]
    eventsForS1 = [p_A1_s1 * p_A2_s1, (1-p_A1_s1) * p_A2_s1, p_A1_s1 * (1-p_A2_s1) ,(1-p_A1_s1) * (1-p_A2_s1)]
    surprisals = map(log, eventsForS1)
    probabilities = [p_A1_s1_speaker * p_A2_s1_speaker, (1-p_A1_s1_speaker) * p_A2_s1_speaker, p_A1_s1_speaker * (1-p_A2_s1_speaker) ,(1-p_A1_s1_speaker) * (1-p_A2_s1_speaker)]
    return sum(map(lambda x : x[0]*x[1], zip(surprisals, probabilities)))

#    print "PROBABILITIES "+str(eventsForS1)
#    return sum(map(lambda x:x * log(x), eventsForS1))
 
print posteriorRestrictionToObjectsAndAdjectivesForSent([0,1,1], True, 0, 1)
print posteriorRestrictionToObjectsAndAdjectivesForSent([1,0,1], True, 1, 0)

def sigma(x):
   return 1/(1+exp(-x))

alpha1 = 1.7
alpha2 = -0.3
u = 0
c = 0
logAvg = -0.6931471805599453
correct = .5
for epoch in range(200):
  for point in prepareNgramData.dataset():
     adj1 = point["AdjNum1"]
     adj2 = point["AdjNum2"]
     pmi1 = point["PMI1"]
     pmi2 = point["PMI2"]
     utilityLoss12 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj1, adj2, 1], True, adj1, adj2)
     utilityLoss21 = posteriorRestrictionToObjectsAndAdjectivesForSent([adj2, adj1, 1], True, adj2, adj1)
     print [utilityLoss12, utilityLoss21, "AGR", agreement[adj1], agreement[adj2]]
     if (utilityLoss12 > utilityLoss21) != (agreement[adj1] < agreement[adj2]):
         print ["ERROR", agreement[adj1], agreement[adj2]]
#     print [utilityLoss12 - utilityLoss21, agreement[adj1] - agreement[adj2]]
#     utilityLoss12 = agreement[adj1] - agreement[adj2]
#     utilityLoss21 = agreement[adj2] - agreement[adj1]

     costLoss12 = -pmi2
     costLoss21 = -pmi1
     u += utilityLoss12 - utilityLoss21
     c += costLoss12 - costLoss21
     logit = alpha1 * (utilityLoss12 - utilityLoss21) + alpha2 * (costLoss12 - costLoss21)
     probability = sigma(logit)
  
 #    print [point["Adj1"],point["Adj2"],utilityLoss12, utilityLoss21, pmi1, pmi2, logit, probability]
     d1Logit = (utilityLoss12 - utilityLoss21)
     d2Logit = (costLoss12 - costLoss21)
     outerDerivative = (1-probability)
     grad1 = outerDerivative * d1Logit
     grad2 = outerDerivative * d2Logit
     alpha1 += .001 * grad1
     alpha2 += .001 * grad2
     print "\t".join(map(str,[alpha1, alpha2, exp(-logAvg), correct]))
     logAvg = 0.99 * logAvg + 0.01 * log(probability)
     correct = 0.99 * correct + 0.01  * (1 if logit > 0 else 0)

