// webppl 19-base.js   --require webppl-viz underscore

// Posterior inference for parameters -- so far only toy code for a single observed data point.
// Can make the number of adjectives larger to scale to entire corpus, and only sampling a single set of worlds

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

// the prior marginal probability of any judgment A(s,x)
var C = [0.2, 0.2, 0.2, 0.2]
 
// the between-speaker correlations by adjectives, between-speaker agreement
// Adjective 1 is more "subjective", adjective 2 is more "objective"
var agreement = [.3, .9, .3, .9]

//// Loss probabilities: lossProb2 should be larger than lossProb1.
//// the probability that the word two words before the end of the current prefix is lost
//var lossProb2 = .5
//// the probability that the second-to-last word in the current prefix is lost
//var lossProb1 = .0 // setting to zero to make the posterior easier to interpret. Any low value is fine though.
//// we could also allow the last word in the current prefix to be lost

// any positive value would do
//var rationality = 1.0

/////////////////////////////////////////////////////

// Given an inter-speaker correlation, samples truth values for judgments A(s1,x), A(s2,x)
// The definition guarantees that the Pearson correlation between the two elements of the return value is the given correlation
var prior_adj = function(adjective) {
   return function(x) {    
                           var j1 = flip(C[adjective]) // A(s1,x)
                           var j2 = agreement[adjective] * (j1 ? 1 : 0) + (1-agreement[adjective]) * C[adjective]; // A(s2,x)
                           return [j1, flip(j2)] };
}

var world_samples = 500;

// Samples from the prior over worlds
// Worlds are encoded as 3D arrays of truth values indexed as follows: world[Adjective][Object][Speaker] holds the truth value for the judgment Adjective(Speaker, Object)
var world_prior = map(function() {
 return [map(prior_adj(0), _.range(n_obj)),map(prior_adj(1), _.range(n_obj)),map(prior_adj(2), _.range(n_obj)),map(prior_adj(3), _.range(n_obj))];
}, _.range(world_samples))


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
var corrupt = function(utterance, lossProb2, lossProb1) {
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
var third = cache(function(prefix, lossProb2, lossProb1) {
   Infer({method : 'enumerate', // samples:1000, incremental:true,
       model() {
//       var world = world_prior(agreementTable)
       var worldIndex = sample(RandomInteger({n: world_samples}))
       var world = world_prior[worldIndex]

      var corruption1 = corrupt([prefix[0]], lossProb2, lossProb1)
      var compatible1 = compatible_utterances(corruption1)
      var compatibleSatisfying1 = any(function(x) { return meaning(x, world, 0) }, compatible1)
      factor(compatibleSatisfying1 ? 0 : -Infinity)

      var corruption2 = corrupt([prefix[0], prefix[1]], lossProb2, lossProb1)
      var compatible2 = compatible_utterances(corruption2)
      var compatibleSatisfying2 = any(function(x) { return meaning(x, world, 0) }, compatible2)
      factor(compatibleSatisfying2 ? 0 : -Infinity)

      var corruption3 = corrupt([prefix[0], prefix[1], prefix[2]], lossProb2, lossProb1)
      var compatible3 = compatible_utterances(corruption3)
      var compatibleSatisfying3 = any(function(x) { return meaning(x, world, 0) }, compatible3)
      factor(compatibleSatisfying3 ? 0 : -Infinity)
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

//var computeMarginalAdj = function(utterance) {
//  return map(function(adj) { return computeMarginalObj(utterance, adj, 0)}, _.range(n_adj))
//}


//////////////////////////////////////////////////////////////////////////
// Inspect the posterior by looking at the coordinate-wise marginals
//////////////////////////////////////////////////////////////////////////

var prettyPrint = function(utterance) {
   var adj1 = "Adjective"+utterance[0]
   var adj2 = "Adjective"+utterance[1]
   var noun = "Noun"+utterance[2]
   return adj1+" "+adj2+" "+noun 
}

//var marginalTable = computeMarginalAdj(options[0])
//var marginalTable2 = computeMarginalAdj(options[1])
//
//


//var generateAgreementTable = cache(function(agreement1, agreement2) {Infer({method : 'rejection', samples : 1, model () {
//              var agr3 = sample(Uniform({a:0,b:1}))
//              var agr4 = sample(Uniform({a:0,b:1}))
//              return [agreement1, agreement2, agr3, agr4]
//            }})})
//
//console.log(generateAgreementTable(0.1, 0.7))

// Distribution over listener / third-party beliefs about the object
// Given the adjectives A1, A2 and the object o given in the sentence, returns
// the joint distribution of A1(s2, o) and A2(s2, o).
// Here, while s1 is the speaker of the utterance, s2 is the belief of another speaker
// (or possibly of the listener, depending on the interpretation of the model.)
var posteriorRestrictionToObjectsAndAdjectivesForSent = cache(function(sentence, lossProb2, lossProb1) {
     Infer({method: 'enumerate', //'rejection', samples : 10, incremental:true,
            model() {
               var model = sample(third(sentence, lossProb2, lossProb1))
               return [model[0][1][1], model[1][1][1]];
           }})})

//console.log(restrictionToObjectsAndAdjectivesForSent(options[0]))
//console.log(restrictionToObjectsAndAdjectivesForSent(options[1]))





   // The speaker chooses a sentence so a to minimize entropy of the posterior listener / third-party belief
   // I'm assuming that the rationality parameter is 1 -- any positive value would do.
   var speaker = cache(function(rationality, lossProb2, lossProb1) {
                      var speakerModel = Infer({method : 'enumerate',
                          model() {
                          var sentence = options[sample(RandomInteger({n : 2}))];
                          //console.log(posteriorRestrictionToObjectsAndAdjectivesForSent(sentence, lossProb2, lossProb1))
                          factor(-rationality*entropy(posteriorRestrictionToObjectsAndAdjectivesForSent(sentence, lossProb2, lossProb1)))
                          return sentence;
                       }});
                       return Gaussian({mu : speakerModel.getDist()['[0,1,1]'].prob, sigma : 0.05})
                     })
   
//   console.log("\nSpeaker Distribution: Subjective adjective is preferred earlier.")
//   console.log(prettyPrint([0,1,1])+"   "+speaker.getDist()['[0,1,1]'].prob);
//   console.log(prettyPrint([1,0,1])+"   "+speaker.getDist()['[1,0,1]'].prob);

var applyToAdjectivePairs = function(rationality, lossProb1, lossProb2, pairs, ratings, index) {
  if(pairs.length == index) {
     return
  } else {
//     console.log("APPLIED")
     console.log(rationality)
//     console.log(ratings[index])
     console.log(lossProb2)
//     console.log(speaker(rationality, lossProb2, lossProb1))
     observe(speaker(rationality, lossProb2, lossProb1), ratings[index])

//     console.log(speaker(rationality, pairs[index][0], pairs[index][1], lossProb2, lossProb1))
//     observe(speaker(rationality, pairs[index][0], pairs[index][1], lossProb2, lossProb1), ratings[index])
     applyToAdjectivePairs(rationality, lossProb1, lossProb2, pairs, ratings, index+1)
  }
}

var computeOrderPreferences = Infer({method : 'MCMC', samples : 500, model() { 
   var rationality = sample(Gaussian({mu : 0, sigma : 10}))
   var lossProb2 = sample(Uniform({a:0, b:1}))
   var lossProb1 = 0.0
   
   applyToAdjectivePairs(rationality, lossProb1, lossProb2, [[0.2,0.5]],[.8], 0)
   return [rationality, lossProb2]
}})

console.log(computeOrderPreferences)
viz.auto(computeOrderPreferences)

1



