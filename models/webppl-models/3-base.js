// webppl base.js --require webppl-viz underscore


var n_adj = 2;
var n_obj = 2;
//var n_aff = 1;
var n_speaker = 2;



var C = .5;

var p_aff_1 = .95 //Uniform({a:.8,b:1.0})
var p_aff_2 = .5 //Uniform({a:.4,b:.6})
var keep_rate = 0.9
var agreement_1 = 0.2
var agreement_2 = 0.9

var p_aff = [p_aff_1, p_aff_2];


//p_aff
//console.log(p_aff);

//var affect = _.sample([0,1])

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



var agreement = [agreement_1, agreement_2] //[Math.random(),Math.random(),Math.random(),Math.random(),Math.random(),Math.random(),Math.random()]




var prior_judgment = function(adjective, object) {
   var j1 = flip(C)
   var j2 = agreement[adjective] * (j1 ? 1 : 0) + (1-agreement[adjective]) * .5;
   return [j1, flip(j2)];
}

//console.log("PRIOR")
//console.log(prior_judgment(1,0,0));
//console.log("---------")

var chosenObject = _.sample(_.range(n_obj)); //Discrete({ps : [1,1,1,1,1,1]})


console.log(chosenObject)

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







var distJudgmentPrior = Infer(
  {method: 'rejection',
  model() { var judgmentSubj = prior_judgment(0,0,chosenObject);
            var judgmentObj =  prior_judgment(0,1,chosenObject);
            var affect = affect_prior() ? 1 : 0
            return [judgmentSubj,judgmentObj, affect]}});
//viz.auto(distJudgmentPrior)


console.log("ENTROPY PRIOR")
console.log(entropy(distJudgmentPrior))


var distJudgmentStochasticObjective = Infer(
  {method: 'rejection', samples:500,
  model() { var judgmentObj =  flip(1-keep_rate*keep_rate) ? posterior_judgment_without_object(0,1,chosenObject) : posterior_judgment(0,1,chosenObject);
            var judgmentSubj = flip(1-keep_rate) ? posterior_judgment_without_object(0,0,chosenObject) : posterior_judgment(0,0,chosenObject);
            var affect = affect_meaning(0)
            return [judgmentSubj,judgmentObj, affect]}});
viz.auto(distJudgmentStochasticObjective)



var distJudgmentStochasticSubjective = Infer(
  {method: 'rejection',samples:500,
  model() { var judgmentSubj = flip(1-keep_rate*keep_rate) ? posterior_judgment_without_object(0,0,chosenObject) : posterior_judgment(0,0,chosenObject);
            var judgmentObj =  flip(1-keep_rate) ? posterior_judgment_without_object(0,1,chosenObject) : posterior_judgment(0,1,chosenObject);
            var affect = affect_meaning(0)
            return [judgmentSubj,judgmentObj, affect]}});
viz.auto(distJudgmentStochasticSubjective)


var gainObjective = (entropy(distJudgmentPrior)-entropy(distJudgmentStochasticObjective))
var gainSubjective = (entropy(distJudgmentPrior)-entropy(distJudgmentStochasticSubjective))

//console.log("ENTROPY")
//console.log(entropy(distJudgmentStochastic))
console.log("ENTROPY GAIN (OBJECTIVE FIRST)  "+gainObjective)
console.log("ENTROPY GAIN (SUBJECTIVE FIRST) "+gainSubjective)
console.log(gainSubjective-gainObjective)
1

