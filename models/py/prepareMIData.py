def readTSV(path, header, sep="\t"):
   with open(path, "r") as inFile:
      data = map(lambda x:x.split(sep), inFile.read().strip().split("\n"))
      if header:
         header = map(lambda x:x.replace('"',''), data[0])
         data = data[1:]
      else:
         header = map(str,range(len(data[0])))
   return (header, data)

dataCooc = readTSV("/afs/cs.stanford.edu/u/mhahn/scr/qp/adj-noun-cooccurrences-books-5.tsv", True) # this one has more data than those numbered 1-4, but there might be larger files still?
dataNouns = readTSV("/afs/cs.stanford.edu/u/mhahn/scr/qp/noun-occurrences-books-5.tsv", True)
dataAdj = readTSV("/afs/cs.stanford.edu/u/mhahn/scr/qp/adj-occurrences-books-5.tsv", True)

nounsCounts = dict(map(lambda x:(x[dataNouns[0].index("Noun")], int(x[dataNouns[0].index("Count")])), dataNouns[1]))
adjsCounts = dict(map(lambda x:(x[dataAdj[0].index("Adjective")], int(x[dataAdj[0].index("Count")])), dataAdj[1]))
pairsCounts = dict(map(lambda x:((x[dataCooc[0].index("Adjective")], x[dataCooc[0].index("Noun")]), int(x[dataCooc[0].index("Count")])), dataCooc[1]))

from math import log

def getPMIUpToConstant(adj,noun):
  if (adj,noun) in pairsCounts:
     pairCount = pairsCounts[(adj,noun)]+1
  else:
      pairCount = 1
  if adj in adjsCounts:
      adjCount = adjsCounts[adj]+1
  else:
      adjCount = 1
  if noun in nounsCounts:
     nounCount = nounsCounts[noun]+1
  else:
     nounCount = 1
  return log(pairCount) - log(adjCount) - log(nounCount)




