// Referent identification. In this setting, no predictions about ordering is made.


// Different from 10:
// nouns are a type of adjectives, do not point to referents
// Listeners try to identify the referent, by doing inferences over speaker attitudes


// HERE the main driver of the effect is FAULTLESS DISAGREEMENT

// for KL, consider https://github.com/mhtess/webppl-oed/blob/master/src/oed.wppl


// webppl 11-base.js --require webppl-viz underscore


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

var meaning = function(utterance, world, person) {
     var denotation = filter(function(x) { world[utterance[0]][x][person] && world[utterance[1]][x][person] && world[utterance[2]][x][person]}, _.range(n_obj))
     return (denotation.length > 0 ? 1 : 0)
}

var corrupt = function(utterance) {
   if(utterance.length == 3) {
       if(flip(.9)) {
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


var is_named = function(utterance, world, person, entity) {
   return world[utterance[0]][entity][person] && world[utterance[1]][entity][person] && world[utterance[2]][entity][person]
}

// The other setting. Here, speaker and listener each have their picture of the world, and they only partially correlate.
// Listener tries to identify object named by the speaker
// this would need to be changed, listener needs to make (basic) inference about what the speaker meant

// This probably also requires nouns? (treat them like very objective adjectives)

// How about:
// - sample a world
// - sample a conforming utterance
// - sample a conforming (for the speaker) entity

// Then: listener obtains a distribution over entities: look at posterior distribution over speaker attitudes
console.log("..............::")
var real_world = world_prior()
var speaker_entity = sample(RandomInteger({n : n_obj}))
var compatible_utterances_for_entity = filter(function(x) { is_named(x, real_world, 0, speaker_entity) }, utterances)
console.log("World")
console.log(real_world)
console.log("Possible utterances")
console.log(compatible_utterances_for_entity)
var utterance = compatible_utterances_for_entity[sample(RandomInteger({n : compatible_utterances_for_entity.length}))]
console.log(utterance)

var firstL = cache(function(prefix) {
   Infer({method : 'rejection', samples:1000, incremental:true,
       model() {
      var entity = sample(RandomInteger({n : n_obj}))
      var corruption1 = corrupt([prefix[0]])
      var compatible1 = compatible_utterances(corruption1)
      var compatibleSatisfying = any(function(x) { return is_named(x, real_world, 1, entity) }, compatible1)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return entity;
     }})})


var secondL = cache(function(prefix) {
   Infer({method : 'rejection', samples:1000, incremental:true,
       model() {
      var entity = sample(firstL([prefix[0]]))
      var corruption2 = corrupt([prefix[0], prefix[1]])
      var compatible2 = compatible_utterances(corruption2)
      var compatibleSatisfying = any(function(x) { return is_named(x, real_world, 1, entity) }, compatible2)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return entity;
     }})})



var thirdL = cache(function(utterance) {
   Infer({method : 'rejection', samples:1000, incremental:true,
       model() {
      var entity = sample(secondL([utterance[0], utterance[1]]))
      var corruption3 = corrupt([utterance[0], utterance[1], utterance[2]])
      var compatible3 = compatible_utterances(corruption3)
      var compatibleSatisfying = any(function(x) { return is_named(x, real_world, 1, entity) }, compatible3)
      factor(compatibleSatisfying ? 0 : -Infinity)
      return entity;
     }})})

var listenerObjects = thirdL(utterance)
console.log("Speaker intended")
console.log(speaker_entity)
console.log("Listener identified")
console.log(listenerObjects)




1



