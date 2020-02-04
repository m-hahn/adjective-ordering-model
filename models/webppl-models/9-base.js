// CURRENTLY this version is in use


// HERE the main driver of the effect is FAULTLESS DISAGREEMENT

// for KL, consider https://github.com/mhtess/webppl-oed/blob/master/src/oed.wppl


// webppl 9-base.js --require webppl-viz underscore


var n_adj = 4;
var n_obj = 2;
//var n_aff = 1;
var n_speaker = 2;



var C = .2;

var agreement_1 = 0.6
var agreement_2 = 0.9

var agreement = [agreement_1, agreement_2]


var prior_judgment = function(adjective, object) {
   var j1 = flip(C)
   var j2 = agreement[adjective] * (j1 ? 1 : 0) + (1-agreement[adjective]) * C;
   return [j1, flip(j2)];
}

var chosenObject = 0 //_.sample(_.range(n_obj)); //Discrete({ps : [1,1,1,1,1,1]})

console.log(chosenObject)


var utterances = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1], [0, 3, 0], [0, 3, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1], [1, 3, 0], [1, 3, 1], [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1], [2, 3, 0], [2, 3, 1], [3, 0, 0], [3, 0, 1], [3, 1, 0], [3, 1, 1], [3, 2, 0], [3, 2, 1], [3, 3, 0], [3, 3, 1]]

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
var meaning = function(utterance, world) {
     if(world[utterance[0]][utterance[2]][0] == false) {
       return 0;
     }
     if(world[utterance[1]][utterance[2]][0] == false) {
       return 0;
     }
     return 1;
}

var corrupt = function(utterance) {
   if(utterance.length == 3) {
       if(flip(.3)) {
           return [-1, utterance[1], utterance[2]]
       }
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
   if(partial[1] != full[1]) {
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

var compatible_utterances = function(partial_utterance) {
   return filter(function(x) { is_compatible(x, partial_utterance) }, utterances)
}

var literalListener = cache(function(utterance) {
   Infer({method: 'rejection',
      model() {
      var world = world_prior()
      var corruption = corrupt(utterance)
      var compatible = compatible_utterances(corruption)
      var index = sample(RandomInteger({n : compatible.length}));
      var full_utterance = compatible[index]
      var m = meaning(full_utterance, world)
      factor(m ? 0 : -Infinity)
      return world;
   }})
})

//console.log(literalListener(utterances[2]));


var first = cache(function(prefix) {
   Infer({method : 'rejection', samples:1000, incremental:true,
       model() {
       var world = world_prior()
      var corruption1 = corrupt([prefix[0]])
      var compatible1 = compatible_utterances(corruption1)
      var compatibleSatisfying = any(function(x) { return meaning(x, world) }, compatible1)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})


var second = cache(function(prefix) {
   Infer({method : 'rejection', samples:1000, incremental:true,
       model() {
       var world = sample(first([prefix[0]]))
      var corruption2 = corrupt([prefix[0], prefix[1]])
      var compatible2 = compatible_utterances(corruption2)
      var compatibleSatisfying = any(function(x) { return meaning(x, world) }, compatible2)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})



var third = cache(function(utterance) {
   Infer({method : 'rejection', samples:1000, incremental:true,
       model() {
       var world = sample(second([utterance[0], utterance[1]]))
      var corruption3 = corrupt([utterance[0], utterance[1], utterance[2]])
      var compatible3 = compatible_utterances(corruption3)
      var compatibleSatisfying = any(function(x) { return meaning(x, world) }, compatible3)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return world;
     }})})


var listenerPosterior = third([0,1,1]) //incrementalLiteralListener([1,1,1])

var marginal1 = function(person, object, adjective) { Infer({method: 'rejection', samples:100,
      model() {
         var world = sample(listenerPosterior)
         return world[adjective][object][person] ? 1 : 0
      }})}

console.log(listenerPosterior);

var computeMarginalPerson = function(adj, obj, person) {
   var result = listMean(map(function(x) { return x.value }, marginal1(person, obj, adj).samples))
   return result
}

var computeMarginalObj = function(adj, obj) {
  if(obj == n_obj) {
    return [];
  } else {
    var first = map(function(person) { return computeMarginalPerson(adj, obj, person) }, _.range(n_speaker))
    var result = [first].concat(computeMarginalObj(adj,obj+1))
    return result
  }
}

var computeMarginalAdj = function(adj) {
  return map(function(adj) { return computeMarginalObj(adj, 0)}, _.range(n_adj))
}

console.log("MARGINALS")
console.log(computeMarginalAdj(0))

1

