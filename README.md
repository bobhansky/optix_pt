# GPU Path Tracing with Nvidia Optix, CUDA

# No source code would be added to this repo as asked.

## Final Demo Imgs
![Contrib](https://github.com/bobhansky/optix_pt/blob/main/dragon_bssrdf.png)

![Contrib](https://github.com/bobhansky/optix_pt/blob/main/dragon_bssrdf_2.png)

## Added Camera movement with progressive display

<img src="https://github.com/bobhansky/optix_pt/blob/main/mis.gif" alt="MIS scene" width="600">

<img src="https://github.com/bobhansky/optix_pt/blob/main/dragon.gif" alt="MIS scene" width="600">

This project implement BSSRDF (Diphole diffusion by Henrik Wann Jensen) with Nvidia Optix 6.5, CUDA 10.1


Theory is from "A Practical Model for Subsurface Light Transport" (2001) by Henrik Wann Jensen et al.

The Implementation is nearly the same as what I did in Lajolla renderer, where I did a cpu implementation of BSSRDF.

All equations below come from the 2 resources above.

# This repo implements a direct light only BSSRDF in path tracing. (indirect light only include specular part)

# BSSRDF Rendering Equation

The rendering equation for BSSRDF is from Jensen et al. [2001]:

```math
L_o(x_o, \omega_o) =
\int_S \int_{H^2}
S(x_i, \omega_i, x_o, \omega_o)
L_i(x_i, \omega_i)
(\mathbf{n}_i \cdot \omega_i)
d\omega_i \, dA_i
```

where:

- $L_o(x_o, \omega_o)$: outgoing radiance at surface point $x_o$ in direction $\omega_o$
- $S(x_i, \omega_i, x_o, \omega_o)$: BSSRDF
- $L_i(x_i, \omega_i)$: incoming radiance at point $x_i$ from direction $\omega_i$
- $\mathbf{n}_i$: surface normal at point $x_i$
- $H^2$: hemisphere of incoming directions above the surface
- $S$: surface of the object

## Diffusion and Single Scattering

Jensen proposes that the complete BSSRDF model is the sum of the diffusion approximation and the single scattering term:

```math
S(x_i, \omega_i, x_o, \omega_o)
=
S_d(x_i, \omega_i, x_o, \omega_o)
+
S^{(1)}(x_i, \omega_i, x_o, \omega_o)
```

where $S_d$ represents the multiple scattering, or diffusion, component, and $S^{(1)}$ represents the single scattering component.

## Multiple Scattering Part

The diffusion approximation is based on the observation that the light distribution in highly scattering media tends to become isotropic, so it can be represented as:

```math
S_d(x_i, \omega_i; x_o, \omega_o)
=
\frac{1}{\pi}
F_t(\eta, \omega_i)
R_d(\|x_i - x_o\|)
F_t(\eta, \omega_o)
```

where:

- $S_d(x_i, \omega_i; x_o, \omega_o)$: diffusion component of the BSSRDF, representing multiple scattering
- $x_i, x_o$: surface positions where light enters $(x_i)$ and exits $(x_o)$ the material
- $F_t(\eta, \omega)$: Fresnel transmission term
- $r = |x_i - x_o|$: distance between the incident and exit points

where $R_d$ is equal to the radiant exitance divided by the incident flux:

```math
R_d(r) =
-D \frac{(\vec{n} \cdot \nabla \phi(x_s))}{d\Phi_i} 
=
\frac{\alpha'}{4\pi}
\left[
z_r(\sigma_{tr} d_r + 1)
\frac{e^{-\sigma_{tr} d_r}}{\sigma_t' d_r^3}
+
z_v(\sigma_{tr} d_v + 1)
\frac{e^{-\sigma_{tr} d_v}}{\sigma_t' d_v^3}
\right]
```

Now $S_d(x_i, \omega_i; x_o, \omega_o)$ is a function of $\omega_i$, $\omega_o$, and $r$.

```math
S_d(x_i, \omega_i; x_o, \omega_o)
=
\frac{1}{\pi}
F_t(\eta, \omega_i)
R_d(\|x_i - x_o\|)
F_t(\eta, \omega_o)
```



## Single Scattering Part

The total outgoing radiance, $L_o^{(1)}$, due to single scattering is

```math
L^{(1)}(x_o, \omega_o)
=
\frac{\sigma_s(x_o)\, F\, p(\omega_i', \omega_o')}{\sigma_t}
e^{-s_i \sigma_t(x_i)}
e^{-s_o \sigma_t(x_o)}
L_i(x_i, \omega_i)
```
$$
G = \frac{\lvert \mathbf{n}_{x_i} \cdot \omega_o' \rvert}
         {\lvert \mathbf{n}_{x_i} \cdot \omega_i' \rvert}
$$
- $F = F_r(x_o, \eta)\, F_r(x_i, \eta)$: Fresnel factor
- $p$: Henyey-Greenstein phase function
- $\omega_i'$: refracted direction of the incoming $L_i$
- $\omega_o'$: refracted direction of the outgoing eye ray
- $G$: geometry factor
- $\sigma_t = \sigma_a + \sigma_s$: extinction coefficient
- $\sigma_{tc} = \sigma_t(x_o) + G\sigma_t(x_i)$: combined extinction coefficient
- $s_i$: distance traveled by the refracted photon inside the medium
- $s_o$: distance traveled by the refracted eye ray inside the medium


# Multiple Importance Sampling 
This is according to  Alan King, Christopher Kulla, Alejandro Conty, and Marcos Fajardo. Bssrdf importance sampling. In ACM SIGGRAPH 2013 Talks, 2013b. doi: 10.1145/2504459.2504520
URL https://pdfs.semanticscholar.org/90da/5211ce2a6f63d50b8616736c393aaf8bf4ca.pdf

In diffusion/multiscattering part, to sample the surface point (so that we calculate r), instead of importance sampling $e^{-\sigma_{tr} d}$, Jensen et al. [2001] (inverse transform sampling for this is hard), I followed https://rendering-memo.blogspot.com/2015/01/bssrdf-importance-sampling-3.html to importance sample $e^{-\sigma_{tr} d^2}$, which is a Gaussian distribution and behaves similarly to the original term.


The MIS for sampling surfce point is like belows:

- At each x_o point, sample a point (2d) on a disk according to gaussian dist.
- Sample the probe ray direction (Normal, tangent, bitanget, this is where the MIS plays in), Project the 2d point to a sphere with radius Rmax using x^2 + y^2 + z^2 = Rmax^2. 
- Test scene intersection with sample ray, and it is only an effective inter if it hits BSSRDF material (itself)
- Calculate MIS weight.  


3 Direction for the probeRay, for details, refer to https://github.com/bobhansky/optix_pt/blob/main/bssrdf_sampling.pdf

The overall idea is :
![Probe ray diagram](https://github.com/bobhansky/optix_pt/blob/main/probeRay1.png?raw=true)

A different probeRay direction:
![Probe ray diagram](https://github.com/bobhansky/optix_pt/blob/main/probeRay2.png?raw=true)

The main idea of MIS weight is "what is the probability that this point was hit if the proberay is shoot along a different axis. (or if the disk sampling process is on a different axis)"


# Overall BSSRDF Pipeline

**For the diffusion/multiple-scattering part:**

1. For a PathVertex with BSSRDF material, sample the probe ray.
2. Find intersections of this probe ray with the BSSRDF object.
3. Use next event estimation to compute the direct illumination and evaluate $S(x_i, \omega_i, x_o, \omega_o)$.
4. For sampled points, calculate the MIS weight, and finally estimate with Monte Carlo by dividing by the relevant PDFs (`lightPdf`, `sampleAxisPdf`, `sampleDiskPdf`).

**For the single-scattering part:**

1. Go inside the BSSRDF object with the refracted ray.
2. Sample distance $t$ for traveling inside the object like we did in volumetric path tracing.
3. At the position after traveling inside the object, sample the light.
4. Propagate the ray along $\mathrm{vec3}(\mathrm{lightPos} - \mathrm{vertPosition})$ to find the exit point on the BSSRDF surface.
5. Trace a shadow ray to test whether the light path is occluded; if not, evaluate Single scattering part.

**For aesthetic reasons**, I followed Jensen et al. [2001] by adding a specular reflection term to make the BSSRDF object look better for certain materials such as jade. I evaluate the specular mirror reflection contribution and apply the Fresnel term $F_r$ 

# Split Each Contribution
I used parameter: 
brdf bssrdf

sigma_a 0.23 0.62 0.38

sigma_s 0.85 0.95 1.0

eta 1.3

g 0.2

for this contribution img, SPP = 16, with probeRay Sample = 16 

![Contrib](https://github.com/bobhansky/optix_pt/blob/main/dragon_compare.png)
