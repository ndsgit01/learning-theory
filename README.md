# learning-theory

boolean_conjunction_prediction.py
- implements functions required to predict the formula of conjunction of boolean literals from training data, measure / estimate errors related to the same
- Ref: https://cs.nyu.edu/~mohri/mlbook/ [second edition, Example 2.6 (Conjunction of Boolean literals)], modified slightly to also allow detecting inability to predict a well defined hypothesis from avalable samples.

rademacher_complexity.py
- implements function to find rademacher complexity of a set and rademacher complexity of a function class
- Ref: https://en.wikipedia.org/wiki/Rademacher_complexity

online_learning.py
- implements algorithms to predict with expert advice for the on-line learning scenario
  - Halving algorithm
  - Weighted majority algorithm
  - Exponential weighted average algorithm
- also implements a simple single thread on-line scenario simulation class that synchronizes generators - takes in generators for input and labels and ensures that labels can be consumed only after a corresponding input has been consumed.
- Ref: https://cs.nyu.edu/~mohri/mlbook/ [second edition, Section 8.2 (On-Line Learning > Prediction with expert advice)]


