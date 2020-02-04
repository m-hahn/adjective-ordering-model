//BASED ON 12, TURNED INTO SPEAKER MODEL 

// CURRENTLY this version is in use

// A variant could be:
// - nouns are like (very objective) adjectives, so there is only fuzzy referent identification


// Speaker and listener have common objects, speaker wants to communicate attitudes/descriptions.

// HERE the main driver of the effect is FAULTLESS DISAGREEMENT

// for KL, consider https://github.com/mhtess/webppl-oed/blob/master/src/oed.wppl


// webppl 13-base.js --require webppl-viz underscore


var n_adj = 4;
var n_obj = 4;
//var n_aff = 1;
var n_speaker = 2;



var C = .5;

var agreement_1 = 0.3
var agreement_2 = 0.9

var agreement = [agreement_1, agreement_2]


var prior_judgment = function(adjective, object) {
   var j1 = flip(C)
   var j2 = agreement[adjective] * (j1 ? 1 : 0) + (1-agreement[adjective]) * C;
   return [j1, flip(j2)];
}

var chosenObject = 0 //_.sample(_.range(n_obj)); //Discrete({ps : [1,1,1,1,1,1]})

console.log(chosenObject)


//var utterances = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1], [0, 3, 0], [0, 3, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1], [1, 3, 0], [1, 3, 1], [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1], [2, 3, 0], [2, 3, 1], [3, 0, 0], [3, 0, 1], [3, 1, 0], [3, 1, 1], [3, 2, 0], [3, 2, 1], [3, 3, 0], [3, 3, 1]]
var utterances = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 3, 0], [0, 3, 1], [0, 3, 2], [0, 3, 3], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 2, 3], [1, 3, 0], [1, 3, 1], [1, 3, 2], [1, 3, 3], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 1, 3], [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 2, 3], [2, 3, 0], [2, 3, 1], [2, 3, 2], [2, 3, 3], [3, 0, 0], [3, 0, 1], [3, 0, 2], [3, 0, 3], [3, 1, 0], [3, 1, 1], [3, 1, 2], [3, 1, 3], [3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 2, 3], [3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 3, 3]]


// for adjective, for object, for speaker, give 1 or 0

var prior_adj = function(agree) {
   return function(x) {    var j1 = flip(C)
                           var j2 = agree * (j1 ? 1 : 0) + (1-agree) * C;
                           return [j1, flip(j2)] };
}

console.log(_.range(5))

console.log(prior_adj(agreement[0]))
console.log(map(prior_adj(agreement[0]), _.range(n_adj)));

var world_prior = function() {
 return [map(prior_adj(agreement[0]), _.range(n_adj)),map(prior_adj(agreement[1]), _.range(n_adj)),map(prior_adj(agreement[0]), _.range(n_adj)),map(prior_adj(agreement[1]), _.range(n_adj))];
}

// for complete utterances without loss
var meaning = function(utterance, world, person) {
     if(world[utterance[0]][utterance[2]][person] == false) {
       return 0;
     }
     if(world[utterance[1]][utterance[2]][person] == false) {
       return 0;
     }
     return 1;
}

var corrupt = function(utterance) {
   if(utterance.length == 3) {
       var corruptFirst = flip(.9)
       var corruptSecond = flip(.1)
       var entry1 = (corruptFirst ? -1 : utterance[0])
       var entry2 = (corruptSecond ? -1 : utterance[1])
       return [entry1, entry2, utterance[2]]
//       if(flip(.9)) {
//           return [-1, utterance[1], utterance[2]]
//       }
   }
   else if(utterance.length == 2) {
       var corruptFirst = flip(.1)
       var entry1 = (corruptFirst ? -1 : utterance[0])
       return [entry1, utterance[1]]
   }
   return utterance
}

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

var compatible_utterances = cache(function(partial_utterance) {
   return filter(function(x) { is_compatible(x, partial_utterance) }, utterances)
})

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


var second = cache(function(prefix) {
   Infer({method : 'enumerate', //samples:1000, incremental:true,
       model() {
       var world = sample(first([prefix[0]]))
      var corruption2 = corrupt([prefix[0], prefix[1]])
      var compatible2 = compatible_utterances(corruption2)
      var compatibleSatisfying = any(function(x) { return meaning(x, world, 0) }, compatible2)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})



var third = cache(function(utterance) {
   Infer({method : 'enumerate', //samples:1000, incremental:true,
       model() {
       var world = sample(second([utterance[0], utterance[1]]))
      var corruption3 = corrupt([utterance[0], utterance[1], utterance[2]])
      var compatible3 = compatible_utterances(corruption3)
      var compatibleSatisfying = any(function(x) { return meaning(x, world, 0) }, compatible3)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})

//var sentence = [1,0,1]
//var listenerPosterior = third(sentence) //incrementalLiteralListener([1,1,1])




var marginal1Enum = cache(function(utterance, person, object, adjective) { Infer({method: 'enumerate',
      model() {
         var world = sample(third(utterance))
         return world[adjective][object][person] ? 1 : 0
      }})})

var marginal1 = cache(function(person, object, adjective) { Infer({method: 'rejection', samples:1000,
      model() {
         var world = sample(listenerPosterior)
         //console.log(world);
         //console.log(world[adjective][object][person]);
         return world[adjective][object][person] ? 1 : 0
      }})})

//console.log("Listener's posterior over worlds:")
//console.log(listenerPosterior);

// this is the setting where speaker and listener have a common referent set and referents can be identified. The speaker wants to communicate knowledge about the object. The interpretation of person 1 might be third persons.
var listenerAboutObject = function(sentence) {
      Infer({method: 'enumerate', //samples : 100, incremental:true,
      model() {
          var model = sample(third(sentence))
          if(meaning(sentence, model, 1)) {
             return 1
          } else {
             return 0
          }
     }})}
//console.log("Listener's belief about person 1's judgment about the object:")
//console.log(listenerAboutObject.getDist()['1'].prob)



//
//var listenerChoosesCorrectObject = Infer({method : 'rejection', samples : 100, incremental:true,
//    model() {
//        var model = sample(listenerPosterior)
//        var object = sample(RandomInteger({n : n_obj}))
//        factor(meaning([sentence[0], sentence[1], object], model, 1) ? 0 : -Infinity)
//        return (object == sentence[2])
//    }})
//console.log(listenerChoosesCorrectObject)



//////////////////////////////////////////////////////////////////////////
// Inspect the posterior by looking at the coordinate-wise marginals
//////////////////////////////////////////////////////////////////////////

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

var options = [[1,0,1], [0,1,1]]



console.log("MARGINALS: probability that a speaker attributes a property to an object")
console.log(options[0])
var marginalTable = computeMarginalAdj(options[0])
console.log("Adjective 1")
console.log(marginalTable[0])
console.log("Adjective 2")
console.log(marginalTable[1])
console.log("Adjective 3")
console.log(marginalTable[2])
console.log("Adjective 4")
console.log(marginalTable[3])
console.log("Key from outer to inner: object - person")


console.log("#########################")
console.log("MARGINALS: probability that a speaker attributes a property to an object")
console.log(options[1])
var marginalTable2 = computeMarginalAdj(options[1])
console.log("Adjective 1")
console.log(marginalTable2[0])
console.log("Adjective 2")
console.log(marginalTable2[1])
console.log("Adjective 3")
console.log(marginalTable2[2])
console.log("Adjective 4")
console.log(marginalTable2[3])
console.log("Key from outer to inner: object - person")



// the distribution specifically for the specific object and two adjectives and two speakers

// TODO there seems to be a problem in the definition, why does it make reference to object zero?
var restrictionToObjectsAndAdjectives = cache(function(speaker) {Infer({method: 'enumerate', //samples : 100, incremental:true,
      model() {
          var model = sample(listenerPosterior)
          return [model[0][0][speaker], model[0][1][speaker], model[1][0][speaker], model[1][1][speaker]];
     }})})

//console.log(restrictionToObjectsAndAdjectives)
//console.log(entropy(restrictionToObjectsAndAdjectives(0)))
//console.log(entropy(restrictionToObjectsAndAdjectives(1)))

//var generalMarginalEntropy =-sum(map(function(x) { return sum(map(function(x) { return sum(map(function(x) { return (x == 1 || x == 0) ? 0 : x*Math.log(x) + (1-x)*Math.log(1-x) }, x)) }, x))},marginalTable   ))
//console.log("Sum of the Marginal Entropies")
//console.log(generalMarginalEntropy);


var restrictionToObjectsAndAdjectivesForSent = cache(function(sentence,speaker) {Infer({method: 'enumerate', //samples : 100, incremental:true,
      model() {
          var model = sample(third(sentence))
          return [model[0][0][speaker], model[0][1][speaker], model[1][0][speaker], model[1][1][speaker]];
     }})})

var speaker = Infer({method : 'enumerate',
                     model() {
                     var sentence = options[sample(RandomInteger({n : 2}))];
                     //var listenerPosterior = third(sentence);
                     factor(-entropy(restrictionToObjectsAndAdjectivesForSent(sentence, 0)))
                     factor(-entropy(restrictionToObjectsAndAdjectivesForSent(sentence, 1)))
                     return sentence;
                    }})

console.log(speaker);

1

// success measures
// - entropy of marginal (not well motivated?)
// - log prob assigned to the meaning of full utterance. but then other speakers not taken into account
// - probability that correct referent will be picked
// - entropy about the specific object and two adjectives, but across speaker and listener (no need to factorize)




