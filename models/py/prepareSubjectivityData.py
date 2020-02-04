import random

def readTSV(path, header, sep="\t"):
   with open(path, "r") as inFile:
      data = map(lambda x:x.split(sep), inFile.read().strip().split("\n"))
      if header:
         header = map(lambda x:x.replace('"',''), data[0])
         data = data[1:]
      else:
         header = map(str,range(len(data[0])))
   return (header, data)

dataSubj = readTSV("/afs/cs.stanford.edu/u/mhahn/qp/corpus-tools/subjectivity.txt", False)

dataSubj = (dataSubj[0], sorted(dataSubj[1], key=lambda x:x[1]))

#print dataSubj[1]
adjToNum = dict(zip(map(lambda x:x[1], dataSubj[1]), range(len(dataSubj[1]))))
agreement = map(lambda x:1-float(x[2]), dataSubj[1])


