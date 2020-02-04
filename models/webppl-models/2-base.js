// webppl base.js --require webppl-viz underscore

// Have speaker and listener models: speaker generates adjectives, listener infers speaker state and speaker judgment


// affect is 0/1, judgment is True/False


// 1 and 2 are subjective
var n_adj = 3;
var n_obj = 2;
//var n_aff = 1;
var n_speaker = 2;

var p_aff = [.5, .2, .8];

var prior_affect = function() {
  return flip(.5) ? 1 : 0
}

var affect = prior_affect()

var utterance_prior = function() {
   return Categorical({ps : [1,1], vs : [[0,1],[0,2]]})
}

// alternative
//var utterancePrior = function() {
//  var utterances = ['some of the people are nice',
//                    'all of the people are nice',
//                    'none of the people are nice']
//  var i = randomInteger(utterances.length)
//  return utterances[i]
//}

var speakerModel = cache(function(affect) {
 Infer({method : 'enumerate', maxExecutions: 2,
        model() { 
                   var utterance = flip(.5)  ? [0,1] : [0,2] // somehow things break when putting Categorical here
                   factor(utterance[1] == 1+affect ? 0 : -1)
                   return utterance
                }
       })
})


// for some reason this doesn't work
var posterior_affect = function(obj, subj) {
    var affect = prior_affect()
    var utterance = speakerModel(affect)
    var compatible = (utterance[0] == obj && utterance[1] == subj)
//    factor(compatible ? 0 : -Infty)
    return affect
}

//console.log(posterior_affect(0,1).score(0))
//console.log(posterior_affect(0,1).score(1))


// distribution over affects
var distJudgmentAffect = Infer(
  {method: 'rejection',
  model() { var affect = posterior_affect(0,1)
            return [affect]}});
viz.auto(distJudgmentAffect)




