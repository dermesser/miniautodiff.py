# miniautodiff

This is not productivity-level code: it's written in order to be understood. Therefore,
just dive into the source!

* `autodiff.py` is a classic, naive reverse-mode algorithm. Gradients are propagated
  in reverse through the expression tree.
* `gad.py` propagates gradients forward through the expression tree. Despite being forward
  mode, gradients w.r.t. each argument are calculated simultaneously. Therefore it will
  scale with the number of function values, not the number of arguments.
