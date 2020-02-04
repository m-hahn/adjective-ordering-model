// CURRENTLY this version is in use


// HERE the main driver of the effect is FAULTLESS DISAGREEMENT

// for KL, consider https://github.com/mhtess/webppl-oed/blob/master/src/oed.wppl


// webppl 8-base.js --require webppl-viz underscore


var n_adj = 2;
var n_obj = 10;
//var n_aff = 1;
var n_speaker = 2;



var C = .2;

var agreement_1 = 0.0 //0.7
var agreement_2 = 0.0 //0.9

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

console.log(prior_adj(agreement[0]))
console.log(map(prior_adj(agreement[0]), [0,1,2,3]));

var world_prior = function() {
 return [map(prior_adj(agreement[0]), [0,1,2,3]),map(prior_adj(agreement[1]), [0,1,2,3]),map(prior_adj(agreement[0]), [0,1,2,3]),map(prior_adj(agreement[1]), [0,1,2,3])];
}

// for complete utterances without loss
var meaning = function(utterance, world) {
     //console.log(world);
     //console.log(utterance)
//     console.log(world[utterance[0]][utterance[2]][0]);
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
       if(flip(.0)) {
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
      //console.log("COMPATIBLE")
      //console.log(compatible)
      //console.log(index);
      //console.log(full_utterance);
      var m = meaning(full_utterance, world)
      factor(m ? 0 : -Infinity)
      return world;
   }})
})

//console.log(literalListener(utterances[2]));


var first = cache(function(prefix) {
   Infer({method : 'rejection', samples:100, incremental:true,
       model() {
       var world = world_prior()
      var corruption1 = corrupt([prefix[0]])
      var compatible1 = compatible_utterances(corruption1)
      var index1 = sample(RandomInteger({n : compatible1.length}));
      var full_prefix1 = compatible1[index1]
      var m1 = meaning(full_prefix1, world)
//      if(true || (m1 ? 0 : -Infinity) == 0) {
//         console.log(full_prefix1)
//         console.log(world[1][0][0])
//         console.log(world[1][1][0])
//      } 
      factor(m1 ? 0 : -Infinity)
      return world;
     }})})


var second = cache(function(prefix) {
   Infer({method : 'rejection', samples:100, incremental:true,
       model() {
       var world = sample(first([prefix[0]]))
      var corruption2 = corrupt([prefix[0], prefix[1]])
 //     console.log(corruption2)
      var compatible2 = compatible_utterances(corruption2)
   //   console.log(compatible2)
      var index2 = sample(RandomInteger({n : compatible2.length}));
      var full_prefix2 = compatible2[index2]
//      console.log(world)
  //    console.log(full_prefix2)
      var m2 = meaning(full_prefix2, world)
      factor(m2 ? 0 : -Infinity)

      return world;
     }})})



var third = cache(function(utterance) {
   Infer({method : 'rejection', samples:100, incremental:true,
       model() {
       var world = sample(second([utterance[0], utterance[1]]))
      var corruption3 = corrupt([utterance[0], utterance[1], utterance[2]])
      var compatible3 = compatible_utterances(corruption3)
      var index3 = sample(RandomInteger({n : compatible3.length}));
      var full_utterance3 = compatible3[index3]
      var m3 = meaning(full_utterance3, world)
      factor(m3 ? 0 : -Infinity)

      return world;
     }})})


// gets a full utterance, performs incremental + lossy computation
//var incrementalLiteralListener = cache(function(utterance) {
//   Infer({method: 'rejection',samples:100,incremental:true,
//      model() {
//      var world = world_prior()
//
//// it is a bit weird that only one completion is sampled in each step, this also causes weird results (adjectives mentioned have relatively high posterior probability even for wrong adjective even when there is zero memory loss)
//
//
//      // first word
//      var corruption1 = corrupt([utterance[0]])
//      var compatible1 = compatible_utterances(corruption1)
//      var index1 = sample(RandomInteger({n : compatible1.length}));
//      var full_utterance1 = compatible1[index1]
//      var m1 = meaning(full_utterance1, world)
////      if(true || (m1 ? 0 : -Infinity) == 0) {
////         console.log(full_utterance1)
////         console.log(world[1][0][0])
////         console.log(world[1][1][0])
////      } 
//      factor(m1 ? 0 : -Infinity)
//
//      // second word
//      var corruption2 = corrupt([utterance[0], utterance[1]])
//      var compatible2 = compatible_utterances(corruption2)
//      var index2 = sample(RandomInteger({n : compatible2.length}));
//      var full_utterance2 = compatible2[index2]
//      var m2 = meaning(full_utterance2, world)
//      factor(m2 ? 0 : -Infinity)
//
//      // third word
//      var corruption3 = corrupt([utterance[0], utterance[1], utterance[2]])
//      var compatible3 = compatible_utterances(corruption3)
//      var index3 = sample(RandomInteger({n : compatible3.length}));
//      var full_utterance3 = compatible3[index3]
//      var m3 = meaning(full_utterance3, world)
//      factor(m3 ? 0 : -Infinity)
//
//      return world;
//   }})
//})
//var listenerPosterior = first([1]) //incrementalLiteralListener([1,1,1])
//var listenerPosterior = second([1,1]) //incrementalLiteralListener([1,1,1])
var listenerPosterior = third([1,1,1]) //incrementalLiteralListener([1,1,1])

var marginal1 = function(person, object, adjective) { Infer({method: 'rejection',
      model() {
         var world = sample(listenerPosterior)
//         console.log("........:::")
//         console.log([person, object, adjective])
//         console.log(world);
//         console.log(world[adjective])
//         console.log(world[adjective][object]);
//         console.log(world[adjective][object][person]);

         return world[adjective][object][person] ? 1 : 0
      }})}

//var   [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [3, 0, 0], [3, 0, 1], [3, 1, 0], [3, 1, 1]]

console.log(listenerPosterior);
//var marg = marginal1(1,1,1)
//console.log(marg)
//console.log(Object.keys(marg))
//console.log(listMean(map(function(x) { return x.value }, marg.samples)))

var computeMarginalPerson = function(adj, obj, person) {
   var result = listMean(map(function(x) { return x.value }, marginal1(person, obj, adj).samples))
//   console.log(result)
   return result
}

var computeMarginalObj = function(adj, obj) {
  if(obj == 2) {
    return [];
  } else {
    var first = map(function(person) { return computeMarginalPerson(adj, obj, person) }, [0,1])
//    console.log(first);
    var result = [first].concat(computeMarginalObj(adj,obj+1))
  //  console.log(result)
    return result
  }
}

var computeMarginalAdj = function(adj) {
  return map(function(adj) { return computeMarginalObj(adj, 0)}, [0,1,2,3])

//  if(adj == 4) {
//    return [];
//  } else {
//    var first = map(function(adj) { return computeMarginalObj(adj, 0)}, [0,1,2,3])
//    console.log("ADJECTIVE "+adj)
//    console.log(first)
//    return [first].concat(computeMarginalAdj(adj+1))
//  }
}

console.log("MARGINALS")
console.log(computeMarginalAdj(0))

1

