import random
import subprocess
import os

files = set(os.listdir("results/"))
ID = 0
while "results/16-base/searchResults_"+str(ID)+".tsv" in files:
   ID += 1

outPath = "results/16-base/searchResults_"+str(ID)+".tsv"
print outPath

with open(outPath, "w") as outFile:
    print >> outFile, "\t".join(["EntropyGain","kappa1","kappa2","kappa3","kappa4","loss_2","loss_1","alpha","C1","C2","C3","C4"])
    
    while True:
      # agr1, agr2 sampled uniformly with constraint agr1 < agr2
      # agr3, agr4 uniformly
      # loss2 sampled uniformly
      # loss1 set to 0.0
      # alpha set to 1.0
      # C same value, uniformly
      while True:  
         agr2 = random.random()
         agr1 = random.random()
         if agr1 < agr2:
            break
      agr3 = random.random()
      agr4 = random.random()
      loss2 = random.random()
      loss1 = 0.0
      alpha = 1.0
      C = random.random()
      proc = subprocess.Popen(['webppl', '16-base.js'] + map(str, [agr1, agr2, agr3, agr4, loss2, loss1, alpha, C, C, C, C]), stdout=subprocess.PIPE)
      entropyGain = proc.stdout.read().strip().split("\n")[-1].replace("\t","_")
      print >> outFile, "\t".join(map(str,[entropyGain, agr1, agr2, agr3, agr4, loss2, loss1, alpha, C, C, C, C]))

