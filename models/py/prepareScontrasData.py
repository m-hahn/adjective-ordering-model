def readTSV(path, header, sep="\t"):
   with open(path, "r") as inFile:
      data = map(lambda x:x.split(sep), inFile.read().strip().split("\n"))
      if header:
         header = map(lambda x:x.replace('"',''), data[0])
         data = data[1:]
      else:
         header = map(str,range(len(data[0])))
   return (header, data)

# TODO maybe instead of the `duplicated' set is the other one the right one? why `duplicated'?
dataRatings = readTSV("scontras-data/order-preference-duplicated.csv", True, ",")
header = dataRatings[0]


def dataset(): #epochs):
    workerCount = 0
    workerDict = {}
    for datapoint in dataRatings[1]:
    #   print "\t".join(datapoint)
       adj1 = datapoint[header.index("correctpred1")].replace('"','')
       adj2 = datapoint[header.index("correctpred2")].replace('"','')
#       if not adj1 in adjToNum or not adj2 in adjToNum:
#          print ["ERROR IN RATINGS DATA", adj1, adj2]
#          continue
       noun = datapoint[header.index("noun")].replace('"','')
       rating = float(datapoint[header.index("correctresponse")])
       worker = datapoint[header.index("workerid")].replace('"','')
       if worker not in workerDict:
         workerDict[worker] = workerCount + 1
         workerCount +=1
       worker = workerDict[worker]
       

#       print [adj1, adj2, adjToNum[adj1], adjToNum[adj2], surpNounC-unigNounC, surpNounI-unigNounI]
       yield {"Adj1" : adj1, "Adj2" : adj2, "Noun" : noun, "worker" : worker, "rating" : rating}

    # don't need to think about the trigram probabilities -- they are a constant factor when comparing orderings
    # also, Adj-Adj bigram probability is irrelavant
    # only need to consider MI between the second adjective and noun
    # But need to do this for both orderings, so need to get dataset where both orderings are ideally in one item
   

