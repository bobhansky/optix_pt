# optix_pt

# No source code would be added to this repo as asked.

## Added Camera movement with progressive display

<img src="https://github.com/bobhansky/optix_pt/blob/main/mis.gif" alt="MIS scene" width="600">

<img src="https://github.com/bobhansky/optix_pt/blob/main/dragon.gif" alt="MIS scene" width="600">

This project implement BSSRDF (Diphole diffusion by Henrik Wann Jensen) with Nvidia Optix 6.5, CUDA 10.1


Implementation follows the blog https://rendering-memo.blogspot.com/2015/01/bssrdf-importance-sampling-1-kickoff.html,

Which is a cpu implementation of "A Practical Model for Subsurface Light Transport" by Henrik Wann Jensen.

All equations come from the above 2 resources.

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


# Finish later...

a demo img (multi scattering only)  is:

<img src="https://github.com/bobhansky/optix_pt/blob/main/multy.png" width="600">

The material parameters aren't tuned, and for now I'm not sure if the implementation is fully correct. Need to check it more before the final ddl.
