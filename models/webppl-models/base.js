// webppl base.js --require webppl-viz underscore


var n_adj = 2;
var n_obj = 2;
//var n_aff = 1;
var n_speaker = 2;

// here, we assume for simplicity that affect is given by [0,1], and an adjective's denotation is a uniform distribution, with length b s.t. differential entropy log(b)? but this has weird properties
// or multinomial
// or gaussian is maybe best: variance depends on the affective informativity
// for now just binomial


var p_aff = [.8, .5];
//var createPAff = function(i) {
//  if(i > 0) {
//     p_aff.push(Math.random())
//     createPAff(i-1)
//  }
//}
//createPAff(n_adj);

p_aff
console.log(p_aff);

var affect = _.sample([0,1])

var prior_affect = function() {
  return flip(.5)
}



var affect_prior = function() {
   return flip(.5);
}



// affective state is a latent variable?
// generate adjective given affect
var affect_meaning = function(adjective) {
  return flip(p_aff[adjective]) ? 1 : 0;
}


//var posterior_affect = function() {
//    var affect = affect_prior()
//    
//
//}

// do inference to figure out the probability assigned by the listener to the correct state of affairs


// referential index resides with the noun

var agreement = [0.5, 0.95] //[Math.random(),Math.random(),Math.random(),Math.random(),Math.random(),Math.random(),Math.random()]

// TODO still not really right 

// create an array containing the judgments

// then memoize a function around it


// random world
// speaker -- adjective -- object
//var judgment_prior = function() {
//   var judgments = []
//   var fill_judgments = function(speaker, adjective, object) {
//     if(speaker == n_speaker) {
//       return;
//     } else if(adjective == n_adj) {
//       fill_judgments(speaker+1, 0, 0)
//     } else if(object == n_obj) {
//       fill_judgments(speaker,adjective+1,0)
//     } else {
//        if(adjective == 0 && object == 0) {
//           judgments.push([])
//        }
//        if(object == 0) {
//            judgments[speaker].push([])
//        }
//        if(speaker == 0) {
//           judgments[speaker][adjective].push(flip(.5))
//        } else {
//          var previous = judgments[0][adjective][object] ? 1 : 0;
//          var prob = agreement[adjective] * previous + (1-agreement[adjective]) * .5;
//          judgments[speaker][adjective].push(flip(prob))
//        }
//        fill_judgments(speaker, adjective, object+1)
//     }
//   }
//   fill_judgments(0,0,0)
//   console.log(judgments);
//   return judgments
//}

//var judgments = judgment_prior()


var prior_judgment = function(adjective, object) {
   var j1 = flip(.5)
   var j2 = agreement[adjective] * (j1 ? 1 : 0) + (1-agreement[adjective]) * .5;
   return [j1, flip(j2)];
}

console.log("PRIOR")
console.log(prior_judgment(1,0,0));
console.log("---------")

var chosenObject = _.sample(_.range(n_obj)); //Discrete({ps : [1,1,1,1,1,1]})
//var chosenAdjective = _.sample(_.range(n_adj)); //Discrete({ps : [1,1,1,1,1,1]})


// TODO write a speaker model


var posterior_judgment_without_object = function(speaker, adjective, object) {
     var judgmentsByS0 = prior_judgment(adjective, 0) // object 0
     var judgmentsByS1 = prior_judgment(adjective, 1) // object 1
     condition(judgmentsByS0[0] || judgmentsByS1[0])
     return (object == 0 ? judgmentsByS0[speaker] : judgmentsByS1[speaker])
}



var posterior_judgment = function(speaker, adjective, object) {
     var judgmentsByS = prior_judgment(adjective, object)
     condition(judgmentsByS[0] == 1)
     return judgmentsByS[speaker]
}

// TODO object vs chosenObject
//var posterior_judgment = function(speaker, adjective, object) {
//     var judgmentsByS = prior_judgment(chosenAdjective, chosenObject)
//     condition(judgmentsByS[0] == 1)
//     return judgmentsByS[1]
////   if(adjective == chosenAdjective && speaker == 0) {
////      return 1;
////   } else if(adjective == chosenAdjective && speaker > 0) {
////      return flip(.5*(agreement[adjective] + prior_judgment(speaker, adjective, object)))
////   } else {
////      return prior_judgment(speaker, adjective, object) // actually, need to cache initial table of decisions
////   }
//}



//var geometric = function() {
//  return flip(.5) ? 0 : geometric() + 1;
//}
//
//var conditionedGeometric = function() {
//  var x = geometric();
//  factor(x > 2 ? 0 : -Infinity);
//  return x;
//}





// infer the listener distribution for affect and, across speakers, judgments for this adjective and object
// from these, compute the probabilities assigned to the actual state of affairs (computing the entropy is enough for now, since we can assume that speaker and listener probabilities are calibrated)


// a distribution over judgments and affects
//var distJudgmentLost = Infer(
//  {method: 'rejection',
//  model() { return [posterior_judgment_without_object(0,0,chosenObject), 
//                    posterior_judgment(0,1,chosenObject)
//                   ] }});
//viz.auto(distJudgmentLost)

var distJudgmentPrior = Infer(
  {method: 'rejection',
  model() { var judgmentSubj = prior_judgment(0,0,chosenObject);
            var judgmentObj =  prior_judgment(0,1,chosenObject);
            var affect = affect_prior() ? 1 : 0
            return [judgmentSubj,judgmentObj, affect]}});
viz.auto(distJudgmentPrior)


//var space = []
//space.push([true,true,0])
//space.push([true,false,0])
//space.push([false,true,0])
//space.push([false,false,0])
//space.push([true,true,1])
//space.push([true,false,1])
//space.push([false,true,1])
//space.push([false,false,1])

//var probsPrior = map(function(x) { return distJudgmentPrior.score(x) }, space)
//var entropyPrior = sum(map(function(x) { return (x == -Infinity) ? 0 : x * Math.exp(x)}, probsPrior))
console.log("ENTROPY PRIOR")
//console.log(entropyPrior)
//console.log("using entropy() function")
console.log(entropy(distJudgmentPrior))




var keep_rate = 0.7

var distJudgmentStochastic = Infer(
  {method: 'rejection',
  model() { var judgmentSubj = flip(1-keep_rate*keep_rate) ? posterior_judgment_without_object(0,0,chosenObject) : posterior_judgment(0,0,chosenObject);
            var judgmentObj =  flip(1-keep_rate) ? posterior_judgment_without_object(0,1,chosenObject) : posterior_judgment(0,1,chosenObject);
            var affect = affect_meaning(0)
            return [judgmentSubj,judgmentObj, affect]}});
viz.auto(distJudgmentStochastic)



//console.log(space)
//console.log(".......:")
//console.log(distJudgmentStochastic.score([false,false,1]))
//console.log(distJudgmentStochastic.score)
//var probs = map(function(x) { return distJudgmentStochastic.score(x) }, space)
//console.log(probs)
//console.log("TO SUM")
//console.log(map(function(x) { return (x == -Infinity) ? 0 : x * Math.exp(x)}, probs))

//var entropyPosterior = sum(map(function(x) { return (x == -Infinity) ? 0 : x * Math.exp(x)}, probs))
console.log("ENTROPY")
//console.log(entropyPosterior)
//console.log("using entropy() function")
console.log(entropy(distJudgmentStochastic))
console.log("ENTROPY GAIN")
console.log(entropy(distJudgmentPrior)-entropy(distJudgmentStochastic))

1
// now compute the uncertainty.



// distribution over affects
//var distAffect = Infer(
//  {method: 'MCMC',
//  model() { return affect_meaning(chosenAdjective) }});
//
//viz.auto(distAffect)


// for each object, and each adjective, compute posterior judgment probability


