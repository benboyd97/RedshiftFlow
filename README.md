# Ben_Boyd_MSc_Project


This repository contains the pipeline for two models designed for rapid galaxy reshift posterior estimation. The first model is a mixture density network (MDN) whilst the second is a normalising flow. The models are trained on simulated data and then applied to real COSMOS20 galaxies which can be downloaded from here: https://cosmos2020.calet.org/ [1]

The pipeline should be used in the follower order:

- simulate data using Simulate_Data.py, the FSPS [2][3] and PROVABGS [4][5] libraries are required for this.
- apply niose to synthetic galaxies using Noise_Model.ipynb
- train models in MDN_COSMOS_Train.ipynb or NSF_COSMOS_Train.ipynb
- the train models can then be applied to held out synthetic data or real COSMOS data in MDN_COSMOS_Train.ipynb or NSF_COSMOS_Train.ipynb. These notebooks also include novel calibrations between synthetic and real datasets.
- comparisons between the two model perfmances can be seen in Model_Comparison.ipynb.


## Design and Novelty

MDN: 

The mixture density network (MDN) was implemented by building
on top of an existing JAX implementation [6]. The original implementation learnt a one-
dimensional probability distribution given a one-dimensional input. The first area of novelty
was to edit the model to allow for multi-dimensional conditionals. The simple neural network
functions were replaced with JAX stax library networks [8], allowing the model to be JIT compatible.

Functions were also added to allow the maximum a posterior (MAP) estimation to be
found from any given number of Gaussian cluster. The mean of each cluster was evaluated in its
respective cluster and all the other clusters, then summed. The mean with the highest summed
probability was given as the MAP prediction. Careful thought was given to these functions to
retain the shapes of the data structures and reduce the number of iterations. This allowed for
the fast MAP prediction for Gaussian Mixtures in a matter of seconds. Further functions were
added to the pipeline to allow for the plotting of the one dimensional multi-cluster posteriors.
Finally, custom integration functions were added to allow the PIT of the mixtures to
be calculated. This was done by summing the weighted cumulative distribution functions of the
mixtures evaluated at the true redshift.

Normalising Flow:

The rational quadratic neural spline flow (NSF), built on top of a
JAX implementation of a single spline layer [7]. The existing implementation only modelled
N-dimensional joint distributions, rather than conditional posteriors. The first modification in-
volved creating neural networks for each spline to make the learnable parameters conditional
on N-dimensional inputs. The spline layers were implemented for N-dimensional distributions
which meant that they needed to be edited to support one-dimensional distributions. Parts of the
Rational Quadratic function needed to be modified to allow for JIT compilation, which requires
all array shapes to be preserved.

There were existing Serial and Flow functions which allowed the stacking of splines into
larger flows [7]. These needed to be edited to ensure that only one-dimension was transformed
and that the original N-dimensional conditionals were injected between each spline.


Further custom functions were added to allow for the MAP prediction to be found from
each flow. Two methods were used to do this. The first method involved sampling the flow
many times and evaluating the probability of each sample. This method was quick, but less
accurate. The second method involved evaluating the flow across a grid and picking the point
with the highest evaluated probability to be the MAP. This probability grid could also be re-used
to plot the posterior as a function of redshift. The grid evaluation function with bin widths of
z = 0.005 was used to determine redshift predictions. A final trapezium numerical integration
used the same gird to evaluate the PIT of each galaxy posterior.


Calibration Implementation:

Since the flux and error offset calibration would involve repeatedly evaluating the JAX models,
the calibration would also need to be made using the same library. This was done using the
JAXopt Scipy Bounded Minimize function [9][10]. Each models respective loss was evaluated 
after applying the calibration parameters. If a parameter was to be left constant during an
optimisation stage, the range of its bound was set to zero. The be application of the calibration
parameters had to be efficient with consistent array shapes. Matrix multiplication was used as
a quick technique for applying calibration parameters to fluxes and errors, before model losses
were evaluated. This also allowed for the optimisation to be JIT compilable for rapid calibration.


## Credits

The simulated data uses the PROABGS stellar population synthesis model [4][5]. The MDN was built on top of an existing implementatioen where extra functionality is added (See Design) [6]. The one dimenisonal conditional normalising flow was adapted from an existing implentation that modelled multi-dimensional joint distributions [7]. 

## References

[1] J. R. Weaver et al., “Cosmos2020: A panchromatic view of the universe to z ∼ 10 from
two complementary catalogs,” ApJS, vol. 258, no. 1, p. 11, 2022.

[2] C. Conroy and J. E. Gunn, “Fsps: Flexible stellar population synthesis,” ASCL, ascl–
1010, 2010.

[3] https://dfm.io/python-fsps/current/

[4] C. Hahn et al., “The desi probabilistic value-added bright galaxy survey (provabgs) mock
challenge,” arXiv preprint arXiv:2202.01809, 2022.

[5] https://github.com/changhoonhahn/provabgs

[6] Hardmare, Mixture density networks with jax, https://github.com/hardmaru/mdn_jax_tutorial, version 816c95e5405522f6214fb0b83a5f741c97ffd4e8, 2020.

[7] C. Waites, Normalizing flows in jax, https://github.com/ChrisWaites/jax-flows, version 26dce814478c656b2ed7e3295ec17b09cad200ee, 2021.

[8] J. Bradbury et al., JAX: Composable transformations of Python+NumPy programs, ver-
sion 0.3.13, 2018. [Online]. Available: http://github.com/google/jax

[9] M. Blondel et al., “Efficient and modular implicit differentiation
,” arXiv preprint arXiv:2105.15183, 2021

[10] https://github.com/google/jaxopt
