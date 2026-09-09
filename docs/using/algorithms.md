# Algorithm details

## Theory of ptychography

Ptychography is an inverse problem where, given a series of diffraction patterns taken under shifted illumination, we attempt to reconstruct the experimental conditions 'most likely' to give us those diffraction patterns.
These diffraction patterns are most often collected on a pixellated camera, but can also be collected with a segmented detector.

In general, we have three variables in ptychography:

 - The 'probe' $P$, a complex field variable indicating the amplitude and phase of the incident wavefunction. In the case of mixed-state ptychography, we have a set of mutually incoherent probes $P_k$, which we call 'probe modes'.
 - The 'object' $O$, which imparts a phase shift and attenuation to the wavefunction. The object can be two dimensional or three dimensional, in which case slices are separated by gaps of thickness $\Delta z_i$. In electron microscopy, the object slices are sometimes referred to as "transmission functions".
 - Probe positions $X_j$. For each position, the probe is shifted to that location and a diffraction pattern taken.

Our forward model is the multislice method, which is capable of modeling interaction with a thick specimen as well as multiple scattering. Given a set of probe modes $P_k$, a 3D object $O_i$, and a probe position $P_j$, we iteratively calculate the wavefunction at each slice given the wavefunction at the slice before:

$$\begin{aligned}
\Psi_{0,k}(\vec{r}) &= P_k(\vec{r} - X_j) \\
\Psi_{i+1,k}(\vec{r}) &= (\Psi_{i,k}(\vec{r}) \cdot O_i(\vec{r})) * p(\Delta z_i)
\end{aligned}$$

In the above, $p(\Delta z)$ indicates a Fresnel free-space propagation kernel of distance $\Delta z$, which is convolved with the wavefunction.

Finally, given the exit wavefunction $\Psi_{n,k}$, we calculate the final intensity in reciprocal space:

$$
I(\vec{k}) = \sum_{k} \left| \mathcal{F}(\Psi_{n, k})(\vec{k}) \right|^2
$$

The inverse problem consists of taking measured diffraction patterns $I_exp$ and recovering the probe $P$ and object $O$.
In the case of single-slice ptychography (where the object is 2D), and the probe positions $P_j$ are perfectly known (and do not fall onto a perfect raster grid), the solution is unambiguous up to a scaling factor of intensity and an affine phase ramp of the object [5].
In practice, the probe positions $P_j$ are not known perfectly, and initial estimates are updated as the algorithm proceeds. This can introduce additional ambiguity, as in the geometrical optics limit, a change in first order aberrations is equivalent to a linear transformation of the probe positions [4].

## Noise models

Ptychography is an overdetermined nonlinear inverse problem. Because the problem is overdetermined, the vast majority of experimental data has no exact solution; any experimental noise whatsoever will likely perturb the problem into this region. This is overcome by the use of maximum likelihood estimation; rather than attempting to find an exact solution, we attempt to find the probe and object 'most likely' to generate the recorded data under some noise model.

The choice of noise model in ptychography has been well-covered in the literature [19,12,18]. Given a modeled intensity $I(\vec{k})$ and a measured intensity $I_{exp}(\vec{k})$, we desire to maximize the probability $P(I | I_{exp})$, i.e. we would like to find the most likely intensity given the experimental data. This is known as the maximum a posteriori estimate. This probability can be obtained using Bayes' theorem:

$$
P(I | I_{exp}) = \frac{P(I_{exp} | I) P(I)}{P(I_{exp})}
$$

However, we usually lack good estimates of the prior probabilities $P(I)$. Using a uniform prior distribution of $P(I)$, maximizing $P(I | I_{exp})$ is equivalent to maximizing the likelihood $P(I_{exp} | I)$:

$$
\max_{I} P(I_{exp} | I) = \max_{I} \prod_{\vec{k}} P(I(\vec{k}) | I_{exp}(\vec{k})) \\
$$

The Gauss-Markov theorem shows that when noise is independent and of constant variance, least squares is the best unbiased linear estimator. We show the maximum-likelihood solution in the case of Gaussian noise of variance $\sigma^2$:

$$\begin{aligned}
P(I | I_{exp}) &= \prod_{\vec{k}} \frac{1}{\sigma \sqrt{2 \pi}} \exp \frac{-(I(\vec{k}) - I_{exp}(\vec{k}))^2}{2 \sigma^2} \\
\mathcal{L}(I) = - \log P(I | I_{exp}) &= \sum_{\vec{k}} \frac{1}{2 \sigma^2} \left( I(\vec{k}) - I_{exp}(\vec{k}) \right)^2 + \log \left( \sigma \sqrt{2 \pi} \right)
\end{aligned}$$

As is customary, we define the loss function $\mathcal{L}(I)$ as the negative log-likelihood. The second term above is a normalization constant and can be ignored.

In the case of Poisson noise, variance is **not** constant, but scales with mean intensity. The maximum likelihood solution is:

$$\begin{aligned}
P(I | I_{exp}) &= \prod_{\vec{k}} \frac{I(\vec{k})^{I_{exp}(\vec{k})} e^{-I(\vec{k})}}{I_{exp}(\vec{k})!} \\
\mathcal{L}(I) &= \sum_{\vec{k}} I(\vec{k}) - I_{exp}(\vec{k}) \log I(\vec{k}) + \log\left(I_{exp}(\vec{k})!\right) \\
\mathcal{L}(I) &\approx \sum_{\vec{k}} \left(I(\vec{k}) - I_{exp}(\vec{k})\right) - I_{exp}(\vec{k}) \left( \log I(\vec{k}) - \log I_{exp}(\vec{k}) \right)
\end{aligned}$$

where at the last step we have applied Stirling's approximation. In practice, a small offset $\epsilon$ must be added to prevent divergences inside the logarithms.
This epsilon can be rationalized as a minimum signal recognizable by the detector. Setting this value high (e.g. 0.1 $e^-$) has the effect of ignoring updates from weak-intensity areas.

When counts are moderate, a variance stabilizing transform can be applied to Poisson distributed data to be approximately Gaussian with a constant variance, allowing the use of a least-squares estimator. This leads to the amplitude and Anscombe noise models. Given a transformation $x \mapsto 2 \sqrt{x + c}$, we can model the transformed variable as Gaussian with variance 1:

$$\begin{aligned}
\mathcal{L}(I) = \sum_{\vec{k}} \frac{1}{2} \left( \sqrt{I(\vec{k}) + c} - \sqrt{I_{exp}(\vec{k}) + c} \right)^2
\end{aligned}$$

$c = 0$ leads to the amplitude noise model, while $c = 3/8$ leads to the Anscombe noise model.
The amplitude and Ascombe noise models have the benefit that additive Gaussian noise can be considered analytically, as discussed by Godard et al. [19].

For the conventional engines, gradients of the loss functions are taken analytically:

$$\begin{aligned}
\nabla \mathcal{L_{p}}(\Psi) &= \left(1 - \frac{I_{exp}(\vec{k})}{\epsilon + I(\vec{k})} \right) \Psi(\vec{k}) \\
\nabla \mathcal{L_{a}}(\Psi) &= \left(1 - \frac{\sqrt{I_{exp}(\vec{k}) + c}}{\epsilon + \sqrt{I(\vec{k}) + c}} \right) \Psi(\vec{k})
\end{aligned}$$

As noted by Leidl et al [18], these two gradients show significant differences in their spatial extent, with the Poisson gradient providing the largest updates at large scattering angles (where signals are weak).

Using these gradients, an optimal step size can be calculated [12], and a total wavefunction update can be found as $\Delta \Psi(\vec{k}) = - \alpha \nabla \mathcal{L}(\Psi)$. For instance, for the amplitude/Anscome loss function:

$$
\Delta \Psi(\vec{k}) = \frac{\sqrt{I_{exp}(\vec{k}) + c}}{\epsilon + \sqrt{I(\vec{k}) + c}} \Psi(\vec{k}) - \Psi(\vec{k})
$$

In the amplitude noise model, this corresponds to the classic modulus constraint of the ePIE method.

## Detailed description of engines

### Gradient descent

The gradient descent engine employs traditional machine learning algorithms to fit the system to the experimental data, minimizing the loss function $\mathcal{L}$.
Autodifferentiation is used to obtain the gradients $\nabla \mathcal{L}$.
Since the loss function $\mathcal{L}$ is a non-constant real function, it is not holomorphic. However, Wirtinger derivatives can be used [13] to overcome this challenge. For real functions, the two Wirtinger derivatives are equivalent:

$$
\overline{\frac{\partial f}{\partial z}} = \frac{\partial f}{\partial \tilde{z}}
$$

And the gradient $\mathbb{R} \to \mathbb{C}$ can be taken as:

$$
\nabla \mathcal{L} = \overline{\frac{\partial \mathcal{L}}{\partial z}}
$$

With the gradient descent engine, solvers can be specified per reconstruction variable in a dictionary.

```yaml
type: 'gradient'

solvers:
  object:
    type: 'sgd'
    learning_rate: 1.0e+0
    momentum: 0.9
  probe:
    type: 'adam'
    learning_rate: 1.0e+1
```

In the above example, the object is updated using stochastic gradient descent (SGD) with Nesterov momentum [20], while the probe is updated using adaptive moment estimation (Adam) [21].
Both of these solvers are commonly used in machine learning.
Another solver option is SGD with a step size given by a Polyak-Ribere conjugate gradient algorithm [22].

Regularizations can be specified in two main forms: Costs which are added to the loss function $\mathcal{L}$ per group, and as constraints which are applied per-group or per-iteration.
Costs are specified in units of electrons—equivalent to one electron on the wrong place of the detector.

Below are some example regularizations:

```yaml
regularizations:  # cost regularizations, applied per group
  - type: obj_l1  # L1 regularization of object
    cost: 15.0

group_constraints:  # constraints, applied per group
  - type: clamp_object_amplitude
    amplitude: 1.1

iter_constraints:  # constraints, applied per iteration
  - type: layers  # low-pass filter object in Z direction
    sigma: 50.0
```

### Conventional engines

Along with the gradient descent engine, phaser implements two conventional ptychography algorithms, ePIE and LSQML. The key difference between the gradient descent engine and the conventional engines is that, in the context of multislice ptychography, the conventional engines form an estimate of the optimized wavefront $\Psi$ at the detector and on each layer. This optimized $\Psi$ is used while calculating the gradient of the previous step. In contrast, in the gradient descent engine, the gradients are all taken simultaneously, and a step is taken in the direciton of the gradient. 

#### ePIE

In the ePIE algorithm, we first use the noise model to compute a wavefront update $\chi(\vec{k})$ on the detector, and backwards propagate it to the exit plane of the sample. Then, at each slice, we split the wavefront update to the object slice and to the previous wavefront/probe:

$$\begin{aligned}
\chi_{n}(\vec{r}) &= \mathcal{F}^{-1}(\chi(\vec{k})) \\
\chi_{i-1} &= \frac{O_i^*(\vec{r})}{\max_{\vec{r}} \left| O_i(\vec{r}) \right|^2} \chi_i(\vec{r}) \\
\Delta O_i &= \frac{P_i^*(\vec{r})}{\max_{\vec{r}} \left| \Psi_i(\vec{r}) \right|^2} \chi_i(\vec{r}) \\
\end{aligned}$$

Finally, we calculate the updates to the probe and object. Probe updates are averaged across the group/batch of positions, while object updates are summed across the group:

$$\begin{aligned}
P(\vec{r}) &\mathrel{+}= \beta_{probe} \frac{\sum_{k} \chi_{0,k}(\vec{r})}{N_k} \\
O_i(\vec{r}) &\mathrel{+}= \beta_{object} \sum_{k} \Delta O_{i,k}(\vec{r})
\end{aligned}$$

This multislice generalization of ePIE (sometimes called 3PIE) was introduced by Maiden et al [8] and is further discussed by Tsai et al [11].
We recognize $O^*(\vec{r})$ as the Wirtinger derivative $\frac{\partial}{\partial \tilde{z}}$ of $P O$ with respect to $P$, showing that single-slice ePIE can be considered a gradient descent method.
For multislice, 3PIE diverges from pure gradient descent in that an update step is taken each slice, prior to the backpropagation of gradients to the previous slice.

In `phaser`, the ePIE solver can be specified as follows:

```yaml
# inside engine
type: 'conventional'
noise_model: 'amplitude'  # for instance

solver:
  type: epie
  # parameters, as described above
  beta_object: 0.1
  beta_probe: 0.1
```

#### LSQML

The LSQML method is implemented as described in Odstrčil et al. [12]. In `phaser`, the LSQML solver can be specified as follows:

```yaml
# inside engine
type: 'conventional'
noise_model: 'amplitude'  # for instance

solver:
  type: lsqml
  # scaling of probe and object updates
  beta_object: 0.1
  beta_probe: 0.1

  # gamma in [12]. eq. 23
  gamma: 0.0

  # delta_P and delta_O in [12] eq. 25
  # illum_reg_object is unitless (it's rescaled by the maximum illumination present)
  illum_reg_object: 0.01
  # illum_reg_probe is unitless
  illum_reg_probe: 0.0
```

#### Regularizations

For the conventional engines, constraint-based regularizations are supported (`group_constraints` and `iter_constraints`) but not cost-based regularizations. 

#### Position solving

For the conventional engines, position solving is performed using the gradient of the loss with respect to a shift in probe position, scaled by the update step size. Three position solvers are supported, a steepest descent solver, a momentum-accelerated solver, and an adaptive momentum solver:

```yaml
type: 'conventional'
solver: ...

position_solver:
  type: 'momentum'  # or 'steepest_descent'
  # can specify a maximum step size, in realspace units
  max_step_size: ~
  # fraction of optimal step size to take
  step_size: 1.0e-2
  # momentum decay rate
  momentum: 0.9
```

`adaptive_momentum` estimates its friction from how quickly recent position updates decorrelate, so consistent drift earns a long memory while oscillation disables momentum:

```yaml
position_solver:
  type: 'adaptive_momentum'
  step_size: 1.0
  max_step_size: ~
  # number of previous iterations to correlate against
  memory: 5
  # multiplier on the accumulated velocity
  gain: 0.5
  # friction = friction_scale * decorrelation rate; smaller -> longer memory
  friction_scale: 0.1
  # friction applied when updates are anticorrelated (momentum disabled)
  oscillation_friction: 0.5
  # skip momentum for positions whose raw update already exceeds this.
  # defaults to `max_step_size`
  momentum_max_update: ~
  # estimate a separate friction for every scan position, rather than one global value
  per_position: false
```

Every continuous parameter above (everything except `memory` and `per_position`) accepts a schedule (`constant`, `piecewise`, or `expr`) as well as a constant, so position updates can be ramped in once the object and probe have settled and tapered off at the end:

```yaml
position_solver:
  type: 'momentum'
  max_step_size: 0.1  # angstrom
  step_size:
    type: 'expr'
    expr: '0.5 * min(1., i / 10.)'
  momentum:
    type: 'piecewise'
    init: 0.0
    steps: {10: 0.9}
```

## References

The following is an incomplete bibliograpy of relevant papers.

### Theory of ptychography

[1] Lin, J. A. & Cowley, J. M. Reconstruction from in-line electron holograms by digital processing. Ultramicroscopy 19, 179–190 (1986).

[2] Rodenburg, J. M. The phase problem, microdiffraction and wavelength-limited resolution — a discussion. Ultramicroscopy 27, 413–422 (1989).

[3] Rodenburg, J. M. Ptychography and Related Diffractive Imaging Methods. in Advances in Imaging and Electron Physics (ed. Hawkes) vol. 150 87–184 (Elsevier, 2008).

[4] Cao, S. Large Field of View Electron Ptychography. (University of Sheffield, 2017).

[5] Fannjiang, A. Raster Grid Pathology and the Cure. Multiscale Model. Simul. 17, 973–995 (2019).


### ePIE

[6] Maiden, A. M. & Rodenburg, J. M. An improved ptychographical phase retrieval algorithm for diffractive imaging. Ultramicroscopy 109, 1256–1262 (2009).

[7] Thibault, P., Dierolf, M., Bunk, O., Menzel, A. & Pfeiffer, F. Probe retrieval in ptychographic coherent diffractive imaging. Ultramicroscopy 109, 338–343 (2009).

[8] Maiden, A. M., Humphry, M. J. & Rodenburg, J. M. Ptychographic transmission microscopy in three dimensions using a multi-slice approach. Journal of the Optical Society of America A 29, 1606 (2012).


### LSQML

[9] Thibault, P. & Guizar-Sicairos, M. Maximum-likelihood refinement for coherent diffractive imaging. New J. Phys. 14, 063004 (2012).

[10] Thibault, P. & Menzel, A. Reconstructing state mixtures from diffraction measurements. Nature 494, 68–71 (2013).

[11] Tsai, E. H. R., Usov, I., Diaz, A., Menzel, A. & Guizar-Sicairos, M. X-ray ptychography with extended depth of field. Opt. Express, OE 24, 29089–29108 (2016).

[12] Odstrčil, M., Menzel, A. & Guizar-Sicairos, M. Iterative least-squares solver for generalized maximum-likelihood ptychography. Optics Express 26, 3108 (2018).


### Gradient descent

[13] Candès, E. J., Li, X. & Soltanolkotabi, M. Phase Retrieval via Wirtinger Flow: Theory and Algorithms. IEEE Transactions on Information Theory 61, 1985–2007 (2015).

[14] Ghosh, S., Nashed, Y. S. G., Cossairt, O. & Katsaggelos, A. ADP: Automatic differentiation ptychography. in 2018 IEEE International Conference on Computational Photography (ICCP) 1–10 (2018). <https://doi.org/10.1109/ICCPHOT.2018.8368470>.

[15] Xu, R. et al. Accelerated Wirtinger Flow: A fast algorithm for ptychography. Preprint at <https://doi.org/10.48550/arXiv.1806.05546> (2018).

[16] Kandel, S. et al. Using automatic differentiation as a general framework for ptychographic reconstruction. Opt. Express, OE 27, 18653–18672 (2019).

[17] Schloz, M. et al. Overcoming information reduced data and experimentally uncertain parameters in ptychography with regularized optimization. Opt. Express, OE 28, 28306–28323 (2020).

[18] Leidl, M. L., Diederichs, B., Sachse, C. & Müller-Caspary, K. Influence of loss function and electron dose on ptychography of 2D materials using the Wirtinger flow. Micron 185, 103688 (2024).


### Regularizations/noise models/solvers

[19] Godard, P., Allain, M., Chamard, V. & Rodenburg, J. Noise models for low counting rate coherent diffraction imaging. Opt. Express, OE 20, 25914–25934 (2012).

[20] Sutskever, I., Martens, J., Dahl, G. & Hinton, G. On the importance of initialization and momentum in deep learning. in Proceedings of the 30th International Conference on Machine Learning 1139–1147 (PMLR, 2013).

[21] Kingma, D. P. & Ba, J. Adam: A Method for Stochastic Optimization. Preprint at <https://doi.org/10.48550/arXiv.1412.6980> (2017).

[22] Loizou, N., Vaswani, S., Laradji, I. & Lacoste-Julien, S. Stochastic Polyak Step-size for SGD: An Adaptive Learning Rate for Fast Convergence. Preprint at <https://doi.org/10.48550/arXiv.2002.10542> (2021).

[23] Tanksalvala, M. et al. Nondestructive, high-resolution, chemically specific 3D nanostructure characterization using phase-sensitive EUV imaging reflectometry. Science Advances 7, eabd9667 (2021).
