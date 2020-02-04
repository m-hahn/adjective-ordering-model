data = read.csv("~/scr/qp/pyro-results/10-predictions-1.tsv", header=TRUE, sep="\t")

library(lme4)

summary(lmer(Rating ~ PMI_Diff + Utility_Diff + (1|Worker), data=data))

summary(lmer(Rating ~ PMI_Diff + Utility_Diff + (1|Worker) + (1|Adj1) + (1|Adj2), data=data))


summary(lmer(Rating ~ PMI_Diff + Utility_Diff + (Utility_Diff|Worker) + (1|Worker), data=data))

summary(lmer(Rating ~ PMI_Diff + Utility_Diff + (Utility_Diff|Worker) + (1|Worker) + (1|Adj1) + (1|Adj2), data=data))



