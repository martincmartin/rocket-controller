# Is on/off/on solver robust for initial burn time less than zero

In the experiments directory has a python script that does on/off/on, i.e. burn/coast/burn, planning for a KSP ascent.  I think it's multi_stage_primer.py but might be a different file.  The docs/ directory holds write ups of various experiments.

There's a test case called `kerbin-coast-first-example` where we don't need a first burn, it's strictly coast/burn.  I'm wondering what happens when we try the on/off/on solver on it.

In particular, does the on/off/on solver allow a negative time for the first burn?  What about negative times for the coast and second burn?  What values does it find for all the parameters it seeks?

What initial values does the code choose for the solver in the kerbin-coast-first-example?  Are they close to the final values?

Essentially, when given a case where we don't need an initial burn, I want to give it to the on/off/on solver, then recognize that no initial burn is needed by tau_{b1} <= 0, so I can later switch to a dedicated off/on solver.  Through code inspection and running the example, I want to make sure the code is robust for this use case.

Don't modify any files outside of tasks/negative-initial-burn, except temporarily as needed for experiments.  Put all build products, including scripts and docs, in the tasks/negative-initial-burn.  Write up your findings in a markdown file in the directory, you can use LaTeX with single $ (inline) or double $$ (display).