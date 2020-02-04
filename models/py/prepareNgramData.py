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

# data = getNgramDataUniteAlsoByCorrect("results-ngrams-books-cleaned-1.csv")
# write.csv(file="/afs/cs.stanford.edu/u/mhahn/scr/qp/results-ngrams-books-cleaned-1_POSTPROCESSED_BY_ITEM.csv", data)

CORPUS = "TRAIN" #TEST" # "TRAIN"
if CORPUS == "TRAIN":
  dataCorpus = readTSV("/afs/cs.stanford.edu/u/mhahn/scr/qp/results-ngrams-books-cleaned-1_POSTPROCESSED_BY_ITEM.csv", True, sep=",")
elif CORPUS == "TEST":
  dataCorpus = readTSV("/afs/cs.stanford.edu/u/mhahn/scr/qp/results-ngrams-books-allAdjs-TEST_PART-9-CLEANED_BY_CORRECT.csv", True, sep=",")

header = dataCorpus[0]
#print header

def dataset(): #epochs):
#  for epoch in epochs:
    random.shuffle(dataCorpus[1])
    for datapoint in dataCorpus[1]:
    #   print "\t".join(datapoint)
       adj1 = datapoint[header.index("Word_C_1")].replace('"','')
       adj2 = datapoint[header.index("Word_C_2")].replace('"','')
       noun = datapoint[header.index("Word_C_3")].replace('"','')

       if not adj1 in adjToNum or not adj2 in adjToNum:
    #      print [adj1, adj2]
          continue
       surpNounC =float(datapoint[header.index("Surp.N.1_C_3")])
       surpNounI =float(datapoint[header.index("Surp.N.1_I_3")])
       unigNounC =float(datapoint[header.index("Surp.N.0_C_3")])
       unigNounI =float(datapoint[header.index("Surp.N.0_I_3")])
       assert unigNounC == unigNounI
#       print [adj1, adj2, adjToNum[adj1], adjToNum[adj2], surpNounC-unigNounC, surpNounI-unigNounI]
       yield {"Adj1" : adj1, "Adj2" : adj2, "AdjNum1" : adjToNum[adj1], "AdjNum2" : adjToNum[adj2], "PMI2" : surpNounC-unigNounC, "PMI1" : surpNounI-unigNounC, 'Noun' : noun}

    # don't need to think about the trigram probabilities -- they are a constant factor when comparing orderings
    # also, Adj-Adj bigram probability is irrelavant
    # only need to consider MI between the second adjective and noun
    # But need to do this for both orderings, so need to get dataset where both orderings are ideally in one item
   

