# Optimizing MLP with Kronecker-factored Approximate Curvature

Here I've written an implementation of the algorithm described in [Martens and Grosse 2015](https://arxiv.org/abs/1503.05671) using only Numpy. Currently, this implementation finds optimal solutions in far less iterations but more time than mini-batch SGD. This is because calculating the Jacobian takes up roughly 80% or more of the time required for each descent step. So in the future I may reimplement this using a Jacobian-vector product engine.
