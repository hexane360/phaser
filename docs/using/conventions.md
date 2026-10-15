# Conventions & glossary

## Units

The core ptychographic routines are unit-independent, so any units can be used. However, since this package is designed primarily for high-resolution electron ptychography, data is typically loaded and stored with length units of Angstrom (\( 1 \:\mathrm{\AA} = 10^{-10} \:\mathrm{m} \)). 
Other units typically follow SI base units.

## Coordinate system

Images are stored row-major, starting with the top-left corner. Keeping with this, points are usually stored as `(y, x)` pairs. Raw data is changed to this convention on import.

A right-handed coordinate system is used. Looking down the optic axis, the x-axis points right, the y-axis points down, and the z-axis points into the page (Forward propagation is the +z direction).

In real-space, the origin is usually centered. In reciprocal space, the origin/zero-frequency point is at the top left corner. The exception is diffraction patterns, which are stored in their original orientation.

Wavefields are stored so total intensity ($I$) is conserved in both spaces:

$$\begin{aligned}
\sum \sum \left| f(x, y) \right|^2 &= I \\
\sum \sum \left| F(k_x, k_y) \right|^2 &= I \\
\end{aligned}$$

Typically, these intensities are kept in units of particles (electrons or photons). This scaling is critical for the Poisson noise model, and keeping the same scaling throughout allows object regularizations to have a stronger effect as dose decreases.

Phase follows the convention where a plane wave is defined as $\exp(2\pi i (\mathbf{k} \cdot \mathbf{r}))$. This is the most common convention, equivalent to defining the Fourier transform as $F(\mathbf{k}) = \int f(\mathbf{r}) \exp(-2\pi i (\mathbf{k} \cdot \mathbf{r})) \:d\mathbf{r}$. However, older crystallography literature uses an opposite sign convention. See Spence and Zuo [1] for more information. A positive defocus aberration is used to indicate overfocus.

The helper functions [`fft2`][phaser.utils.num.fft2] and [`ifft2`][phaser.utils.num.ifft2], as well as [`Sampling`][phaser.utils.num.Sampling], help to enforce these conventions.

[1] Spence, J. C.H. & Zuo, J. M. Electron Microdiffraction. (Plenum Press, New York, 1992).