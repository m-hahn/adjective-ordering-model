// webppl 16-base.js


// the number of adjectives
var n_adj = 4;

// the number of objects
var n_obj = 4;

// the number of speakers
// The utterance whose interpretation is simulated here is uttered by speaker 1
var n_speaker = 2;


// The set of possible utterances: every combination of two adjectives and nouns that is within the bounds given by n_adj (for the adjectives), n_obj (for the nouns)
var utterances = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 3, 0], [0, 3, 1], [0, 3, 2], [0, 3, 3], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 2, 3], [1, 3, 0], [1, 3, 1], [1, 3, 2], [1, 3, 3], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 1, 3], [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 2, 3], [2, 3, 0], [2, 3, 1], [2, 3, 2], [2, 3, 3], [3, 0, 0], [3, 0, 1], [3, 0, 2], [3, 0, 3], [3, 1, 0], [3, 1, 1], [3, 1, 2], [3, 1, 3], [3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 2, 3], [3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 3, 3]]

// The two utterances under considerations: either the subjective adjective comes first, or the objective one 
var options = [[1,0,1], [0,1,1]]



///////////////////////////////////////////////////

// Numerical parameters: Parameters in this section can be changed. As long as the adjectives differ in the inter-speaker correlations (higher correlation for the less `subjective' adjective) and loss probability increases with distance, the effects should come out in the predicted direction.

var agr1 = Number.parseFloat(process.argv[3]);
var agr2 = Number.parseFloat(process.argv[4])
var agr3 = Number.parseFloat(process.argv[5])
var agr4 = Number.parseFloat(process.argv[6])

 
// the between-speaker correlations by adjectives, between-speaker agreement
// Adjective 1 is more "subjective", adjective 2 is more "objective"
var agreement = [agr1, agr2, agr3, agr4]

// Loss probabilities: lossProb2 should be larger than lossProb1.
// the probability that the word two words before the end of the current prefix is lost
var lossProb2 = Number.parseFloat(process.argv[7])
// the probability that the second-to-last word in the current prefix is lost
var lossProb1 = Number.parseFloat(process.argv[8]) // setting to zero to make the posterior easier to interpret. Any low value is fine though.
// we could also allow the last word in the current prefix to be lost

// any positive value would do
var rationality = Number.parseFloat(process.argv[9])


// the prior marginal probability of any judgment A(s,x)
var C = [Number.parseFloat(process.argv[10]), Number.parseFloat(process.argv[11]), Number.parseFloat(process.argv[12]), Number.parseFloat(process.argv[13])]




/////////////////////////////////////////////////////

// Given an inter-speaker correlation, samples truth values for judgments A(s1,x), A(s2,x)
// The definition guarantees that the Pearson correlation between the two elements of the return value is the given correlation
var prior_adj = function(adjective) {
   return function(x) {    
                           var j1 = flip(C[adjective]) // A(s1,x)
                           var j2 = agreement[adjective] * (j1 ? 1 : 0) + (1-agreement[adjective]) * C[adjective]; // A(s2,x)
                           return [j1, flip(j2)] };
}

// Samples from the prior over worlds
// Worlds are encoded as 3D arrays of truth values indexed as follows: world[Adjective][Object][Speaker] holds the truth value for the judgment Adjective(Speaker, Object)
var world_prior = function() {
 return [map(prior_adj(0), _.range(n_obj)),map(prior_adj(1), _.range(n_obj)),map(prior_adj(2), _.range(n_obj)),map(prior_adj(3), _.range(n_obj))];
}

// The meaning function, for complete utterances without loss.
// An utterance uttered by `person' is true in `world' if this person judges both adjectives to apply to the object
var meaning = function(utterance, world, person) {
     // utterance[0], utterance[1] are adjectives
     // utterance[2] is the noun, which we assume uniquely identified the object
     if(world[utterance[0]][utterance[2]][person] == false) {
       return 0;
     }
     if(world[utterance[1]][utterance[2]][person] == false) {
       return 0;
     }
     return 1;
}


// Takes a prefix and randomly replaces earlier words with -1
var corrupt = function(utterance) {
   if(utterance.length == 3) {
       var corruptFirst = flip(lossProb2)
       var corruptSecond = flip(lossProb1)
       var entry1 = (corruptFirst ? -1 : utterance[0])
       var entry2 = (corruptSecond ? -1 : utterance[1])
       return [entry1, entry2, utterance[2]]
   } else if(utterance.length == 2) {
       var corruptFirst = flip(lossProb1)
       var entry1 = (corruptFirst ? -1 : utterance[0])
       return [entry1, utterance[1]]
   }
   return utterance
}

// Checks whether a (possibly corrupted) prefix (e.g., [0,1] or [-1,1,1]) matches a full utterance (e.g., [0,1,1])
var is_compatible = function(full, partial) {
   if(partial[0] != -1 && partial[0] != full[0]) {
      return false;
   }
   if(partial.length == 1) {
      return true
   }
   if(partial[1] != -1 && partial[1] != full[1]) {
      return false;
   }
   if(partial.length == 2) {
      return true
   }
   if(partial[2] != full[2]) {
      return false;
   }
   return true;
}

// Collects and caches utterance compatible with a (possibly corrupted) prefix
var compatible_utterances = cache(function(partial_utterance) {
   return filter(function(x) { is_compatible(x, partial_utterance) }, utterances)
})

// The listener posterior after hearing the first word
var first = cache(function(prefix) {
   Infer({method : 'rejection', samples:1000, incremental:true,
       model() {
       var world = world_prior()
      var corruption1 = corrupt([prefix[0]])
      var compatible1 = compatible_utterances(corruption1)
      var compatibleSatisfying = any(function(x) { return meaning(x, world, 0) }, compatible1)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})

// Listener posterior after hearing the second word
var second = cache(function(prefix) {
   Infer({method : 'enumerate',
       model() {
       var world = sample(first([prefix[0]]))
      var corruption2 = corrupt([prefix[0], prefix[1]])
      var compatible2 = compatible_utterances(corruption2)
      var compatibleSatisfying = any(function(x) { return meaning(x, world, 0) }, compatible2)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})


// Listener posterior after hearing the third word
var third = cache(function(utterance) {
   Infer({method : 'enumerate', 
       model() {
       var world = sample(second([utterance[0], utterance[1]]))
      var corruption3 = corrupt([utterance[0], utterance[1], utterance[2]])
      var compatible3 = compatible_utterances(corruption3)
      var compatibleSatisfying = any(function(x) { return meaning(x, world, 0) }, compatible3)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})

// The marginal posterior probability of the judgment Adjective(Person, Object), after hearing an utterance
var marginal1Enum = cache(function(utterance, person, object, adjective) { Infer({method: 'enumerate',
      model() {
         var world = sample(third(utterance))
         return world[adjective][object][person] ? 1 : 0
      }})})


//////////////////////////////////////////////////////////////////////////
// Inspect the posterior by looking at the coordinate-wise marginals
//////////////////////////////////////////////////////////////////////////

// Recursively create the table of the posterior marginals
var computeMarginalPerson = function(utterance, adj, obj, person) {
   //var result = listMean(map(function(x) { return x.value }, marginal1(person, obj, adj).samples))
   var distribution = marginal1Enum(utterance, person, obj, adj).getDist()
   if(distribution['0'] != undefined && distribution['0']['val'] == 1) {
      return distribution['0']['prob']
   } else {
      return distribution['1']['prob']
   }
   return result
}

var computeMarginalObj = function(utterance, adj, obj) {
  if(obj == n_obj) {
    return [];
  } else {
    var first = map(function(person) { return computeMarginalPerson(utterance, adj, obj, person) }, _.range(n_speaker))
    var result = [first].concat(computeMarginalObj(utterance, adj,obj+1))
    return result
  }
}

var computeMarginalAdj = function(utterance) {
  return map(function(adj) { return computeMarginalObj(utterance, adj, 0)}, _.range(n_adj))
}


//////////////////////////////////////////////////////////////////////////
// Inspect the posterior by looking at the coordinate-wise marginals
//////////////////////////////////////////////////////////////////////////

var prettyPrint = function(utterance) {
   var adj1 = "Adjective"+utterance[0]
   var adj2 = "Adjective"+utterance[1]
   var noun = "Noun"+utterance[2]
   return adj1+" "+adj2+" "+noun 
}

//console.log("MARGINALS: probability that a speaker attributes a property to an object")
//console.log("After hearing utterance "+options[0])
var marginalTable = computeMarginalAdj(options[0])
//console.log("Adjective 1")
//console.log(marginalTable[0])
//console.log("Adjective 2")
//console.log(marginalTable[1])
//console.log("Adjective 3")
//console.log(marginalTable[2])
//console.log("Adjective 4")
//console.log(marginalTable[3])
//console.log("Key from outer to inner: object - person")
//
//console.log("#########################")
//console.log("MARGINALS: probability that a speaker attributes a property to an object")
//console.log("After hearing utterance "+options[1])
var marginalTable2 = computeMarginalAdj(options[1])
//console.log("Adjective 1")
//console.log(marginalTable2[0])
//console.log("Adjective 2")
//console.log(marginalTable2[1])
//console.log("Adjective 3")
//console.log(marginalTable2[2])
//console.log("Adjective 4")
//console.log(marginalTable2[3])
//console.log("Key from outer to inner: object - person")

console.log("Posterior belief of the listener about the judgments of speaker and third-party speaker")
console.log("In these simulations, Adjective0 (A0) is more subjective than Adjective1 (A1).")
console.log("The speaker is S1, while S2 is a third-party speaker (or perhaps the listener, depending on the interpretation of the model).")
console.log("...........")
//console.log("Marginals for the relevant dimensions")
console.log("After hearing Utterance:   "+prettyPrint(options[0]))
console.log("Marginals for relevant dimensions (by adjective, speaker, object)")
console.log( "A0(S1,O1)\t"+marginalTable[0][1][0]+"\t (Posterior that the speaker judges A0 to hold. Should be high because this adjective came second.)")
console.log( "A0(S2,O1)\t"+marginalTable[0][1][1]+"\t (Posterior that third-party speaker judges A0 to hold. Low, due to low inter-speaker correlation.)")
console.log( "A1(S1,O1)\t"+marginalTable[1][1][0]+"\t (Posterior that the speaker judges A1 to hold. Somewhat lower, since this adjective came first and was subject to loss.)")
console.log( "A1(S2,O1)\t"+marginalTable[1][1][1]+"\t (Posterior that the third-party speaker judges A1 to hold)")
console.log("")
console.log("............")
console.log("After hearing Utterance:   "+prettyPrint(options[1]))
console.log("Marginals for relevant dimensions (by adjective, speaker, object)")
console.log( "A0(S1,O1)\t"+marginalTable2[0][1][0]+"\t (Somewhat lower since A0 was subject to loss)")
console.log( "A0(S2,O1)\t"+marginalTable2[0][1][1])
console.log( "A1(S1,O1)\t"+marginalTable2[1][1][0])
console.log( "A1(S2,O1)\t"+marginalTable2[1][1][1]+"\t (High, due to strong inter-speaker correlation)")



//////////////////////////////////////////////////////////////////////////
//  Speaker Model 
//////////////////////////////////////////////////////////////////////////



// Distribution over listener / third-party beliefs about the object
// Given the adjectives A1, A2 and the object o given in the sentence, returns
// the joint distribution of A1(s2, o) and A2(s2, o).
// Here, while s1 is the speaker of the utterance, s2 is the belief of another speaker
// (or possibly of the listener, depending on the interpretation of the model.)
var posteriorRestrictionToObjectsAndAdjectivesForSent = cache(function(sentence) {Infer({method: 'enumerate', //samples : 100, incremental:true,
      model() {
          var model = sample(third(sentence))
          return [model[0][1][1], model[1][1][1]];
     }})})

//console.log(restrictionToObjectsAndAdjectivesForSent(options[0]))
//console.log(restrictionToObjectsAndAdjectivesForSent(options[1]))


// TODO add surprisal of speaker-ground truth to the utility term

// The speaker chooses a sentence so a to minimize entropy of the posterior listener / third-party belief
// I'm assuming that the rationality parameter is 1 -- any positive value would do.
var speaker = Infer({method : 'enumerate',
                     model() {
                     var sentence = options[sample(RandomInteger({n : 2}))];
                     factor(-entropy(posteriorRestrictionToObjectsAndAdjectivesForSent(sentence)))
                     return sentence;
                    }})

console.log("\nSpeaker Distribution: Subjective adjective is preferred earlier.")
console.log(prettyPrint([0,1,1])+"   "+speaker.getDist()['[0,1,1]'].prob);
console.log(prettyPrint([1,0,1])+"   "+speaker.getDist()['[1,0,1]'].prob);

entropy(posteriorRestrictionToObjectsAndAdjectivesForSent(options[0])) - entropy(posteriorRestrictionToObjectsAndAdjectivesForSent(options[1]))


