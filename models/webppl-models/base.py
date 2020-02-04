# adjectives

# parameters:
## for each adjective: inter-speaker agreement
## infomativeness about affective state

import numpy as np

n_adj = 70
n_obj = 50
n_aff = 10

# not etablishing reference, but inference about speaker judgments!

adjectives = range(n_adj)
objects = range(n_obj)
affect = range(n_aff)

agreement = np.random.rand(1,n_adj)
affectInfo = np.random.rand(1, n_aff)


# on each trial, combine object and affect
# have random distributions over properties, subject to speaker agreement constraints
# here, no assumption about dependencies





