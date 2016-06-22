# June 12

1. Adding support for many types of agents (the altruistic type). This will require updating the code in the observe_k function as well as writing a version of sample\_alpha that is more abstract.

1. Adding more complicated games, including multiple games, including sequential games, simultaneous games (that have limited observability). The ultimate version of this is to build a "grammar over games". Stringing games together etc.

1. Implement learning about the person "acted upon". If person 3 knows person 1 is nice (from previous interactions) and person 3 doesn't know anything about person 2 (has never seen him act). If person 1 is mean to person 2, person 3 should infer that person 1 knew something about person 2 even though person 3 didn't see anything. Try capturing this by allowing the prior agents have over other agents to be a random variable and hence learnable. Think about the connection between the probability that an interaction is observed and the parameter of how likely other agents are to have a known value.

1. Also think about doing Gibbs sampling to check our model against (which is not online). 
