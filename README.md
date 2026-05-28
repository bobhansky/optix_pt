# optix_pt

# No source code would be added to this repo as asked.

This project implement BSSRDF (Diphole diffusion by Henrik Wann Jensen) with Nvidia Optix 6.5, CUDA 10.1


Implementation follows the blog https://rendering-memo.blogspot.com/2015/01/bssrdf-importance-sampling-1-kickoff.html,

Which is a cpu implementation of "A Practical Model for Subsurface Light Transport" by Henrik Wann Jensen.


The rendering equation for BSSRDF is from Jensen et al. [2001]:

$$
L_o(x_o, \omega_o)
=
\int_S \int_{H^2}
S(x_i, \omega_i, x_o, \omega_o)
L_i(x_i, \omega_i)
(\mathbf{n}_i \cdot \omega_i)
d\omega_i \, dA_i
$$

where:

- $L_o(x_o, \omega_o)$: outgoing radiance at surface point $x_o$ in direction $\omega_o$
- $S(x_i, \omega_i, x_o, \omega_o)$: BSSRDF
- $L_i(x_i, \omega_i)$: incoming radiance at point $x_i$ from direction $\omega_i$
- $\mathbf{n}_i$: surface normal at point $x_i$
- $H^2$: hemisphere of incoming directions above the surface
- $S$: surface of the object