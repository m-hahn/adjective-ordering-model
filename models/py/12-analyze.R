data = read.csv("~/scr/qp/pyro-12-model-out-1.tsv", sep="\t")
data = data[as.character(data$Adj1) != as.character(data$Adj2),]

data$sigmoid = 1/(1+exp(-data$Logit))



data = read.csv("/afs/cs.stanford.edu/u/mhahn/scr/qp/pyro-12-model-out-2_TEST.tsv", sep="\t")
data = data[as.character(data$Adj1) != as.character(data$Adj2),]
data$sigmoid = 1/(1+exp(-data$Logit))


