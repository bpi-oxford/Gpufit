#ifndef GPUFIT_SPIMACFN_CUH_INCLUDED
#define GPUFIT_SPIMACFN_CUH_INCLUDED

/* Description of the calculate_spim_acfN function
* ============================================================
*
* This function calculates the 3D autocorrelation function (ACF) of Single Plane
* Illumination (SPIM) Fluorescence Correlation Spectroscopy (FCS) and its partial
* derivatives with respect to the model parameters.
*
* Model equation:
*
*   fac   = D * tau + sigma_xy^2
*   u     = a / (2 * sqrt(fac))
*   h(u)  = erf(u) + (exp(-u^2) - 1) / (u * sqrt(pi))
*   argz  = sqrt(1 + D * tau / sigma_z^2)
*   G(tau) = G_inf + h(u)^2 / (2 * sqrt(pi) * N * a^2 * argz)
*
* This is the same equation as the CPU model in core/models.py (SPIM_3D_Free branch),
* confirmed by the identity  gxy = h^2 / a^2  and  gz = 1 / (2*sqrt(pi)*sqrt(argz)),
* with the parameter relationship  GN0 (CPU) = 1 / N (GPU).
*
* See docs/SPIM_ACF_MODEL.md for derivation, conventions, and numerical verification.
*
* NOTE: This replaces the previous Wohland 2010 convention which used 4*D*tau
* instead of D*tau. That convention caused degenerate fits (D -> 2e6 um^2/s)
* because the GPU and CPU models had different functional shapes.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: diffusion coefficient D  (um^2/s)
*             p[1]: number of particles N = 1 / GN0
*             p[2]: long-lag offset G_inf
*             p[3]: lateral PSF 1/e^2 radius sigma_xy  (um)
*             p[4]: axial    PSF 1/e^2 radius sigma_z   (um)
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector of lag times tau (one per data point, in seconds).
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_spim_acfN function
* ====================================================
*
* This __device__ function can be only called from a __global__ function or an
* other __device__ function.
*/

__device__ void calculate_spim_acfN(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // parameters
    REAL const * p = parameters;

    // constants
    REAL const pi      = 3.14159265358979f;
    REAL const sqrt_pi = sqrt(pi);
    REAL const a       = 0.145f; // pixel side length in object space (um)

    // read lag time x = tau from user_info
    REAL * user_info_float = (REAL*) user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin   = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // -----------------------------------------------------------------------
    // Intermediate variables (see docs/SPIM_ACF_MODEL.md §4)
    // -----------------------------------------------------------------------
    REAL const sigma_xy = p[3];
    REAL const sigma_z  = p[4];

    REAL const fac   = p[0]*x + sigma_xy*sigma_xy;          // D*tau + wxy^2
    REAL const u     = a / (2.0f * sqrt(fac));               // a/(2*sqrt(fac))
    REAL const eu2   = exp(-u*u);                            // exp(-u^2)
    REAL const h     = erf(u) + (eu2 - 1.0f)/(u * sqrt_pi); // h(u)
    REAL const argz  = sqrt(1.0f + p[0]*x / (sigma_z*sigma_z)); // sqrt(1 + D*tau/wz^2)
    REAL const A     = 1.0f / (2.0f * sqrt_pi * p[1] * a*a); // 1/(2*sqrt_pi*N*a^2)

    // -----------------------------------------------------------------------
    // Model value
    // G(tau) = G_inf + h^2 / (2*sqrt_pi * N * a^2 * argz)
    //        = G_inf + A * h^2 / argz
    // -----------------------------------------------------------------------
    value[point_index] = p[2] + A * h*h / argz;

    // -----------------------------------------------------------------------
    // Partial derivatives
    //
    // Shared chain-rule pieces:
    //   dh_du    = (1 - eu2) / (u^2 * sqrt_pi)
    //   du_dD    = -a*x    / (4 * fac^(3/2))
    //   du_dwxy  = -a*wxy  / (2 * fac^(3/2))
    // -----------------------------------------------------------------------
    REAL const dh_du   = (1.0f - eu2) / (u*u * sqrt_pi);
    REAL const fac32   = fac * sqrt(fac);                    // fac^(3/2)
    REAL const du_dD   = -a * x          / (4.0f * fac32);
    REAL const du_dwxy = -a * sigma_xy   / (2.0f * fac32);

    REAL * current_derivatives = derivative + point_index;

    // dG/dD
    //   = A * (2*h*dh_du*du_dD / argz  -  h^2 * x / (2*wz^2 * argz^3))
    current_derivatives[0 * n_points] =
        A * (2.0f*h*dh_du*du_dD / argz
             - h*h * x / (2.0f * sigma_z*sigma_z * argz*argz*argz));

    // dG/dN
    //   = -h^2 / (2*sqrt_pi * N^2 * a^2 * argz)
    //   = -A/N * h^2 / argz
    current_derivatives[1 * n_points] = -(A / p[1]) * h*h / argz;

    // dG/dG_inf
    current_derivatives[2 * n_points] = 1.0f;

    // dG/d(sigma_xy)
    //   = A * 2*h*dh_du*du_dwxy / argz
    current_derivatives[3 * n_points] = A * 2.0f*h*dh_du*du_dwxy / argz;

    // dG/d(sigma_z)
    //   = A * h^2 * D*x / (sigma_z^3 * argz^3)
    current_derivatives[4 * n_points] =
        A * h*h * p[0]*x / (sigma_z*sigma_z*sigma_z * argz*argz*argz);
}

#endif
