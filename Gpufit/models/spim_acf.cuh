#ifndef GPUFIT_SPIMACF_CUH_INCLUDED
#define GPUFIT_SPIMACF_CUH_INCLUDED

/* Description of the calculate_spim_acf function
* ============================================================
*
* This function calculates the 3D autocorrelation function (ACF) of Single Plane Illumination (SPIM) 
* Fluorescence Correlation Spectroscopy (FCS) defined as equation 6
*  [Thorsten Wohland, Xianke Shi, Jagadish Sankaran, and Ernst H.K. Stelzer, 
* "Single Plane Illumination Fluorescence Correlation Spectroscopy (SPIM-FCS) 
* probes inhomogeneous three-dimensional environments," 
* Opt. Express 18, 10627-10641 (2010)] and their partial derivatives
* with respect to the model parameters.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: diffusion coefficient D
*             p[1]: number of particles $N =\left \langle C \right \rangle\cdot a^2\cdot 2\sigma_z$, 
*                   where ⟨𝐶⟩ is the average concentration, 
*                   a is the side length of a square pixel in object space, 
*                   and 𝜎𝑧 is the 1/e2 radius of the Gaussian profile in z-direction. 
*             p[2]: convergence value of the ACF for long times G_inf, usually converge to value around 1
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
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The size of user_info in bytes. (not used)
*
* Calling the calculate_spim_acf function
* ====================================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
* reference: https://github.com/ImagingFCS/Imaging_FCS_1_52/blob/main/agpufitjni.cu#L424
*/

__device__ void calculate_spim_acf(
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
    
    // arguments

    REAL const pi = 3.14159f;
    REAL const sqrt_pi = sqrt(pi);
    REAL const a = 0.24; // side length of square pixel in object space, in um
    REAL const sigma_xy = 1.0; // radius of psf in xy plane
    REAL const sigma_z = 1.0; // radius of psf in z plane

    // indices

    REAL * user_info_float = (REAL*) user_info;
    REAL x = 0; // lagged time tau
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
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // value
    REAL const argxy = p[0]*x + pow(sigma_xy,2);
    REAL const argz  = 1.0 + p[0]*x / pow(sigma_z,2);

    REAL const prefix = 4.0*pow(a,2)*sqrt_pi*p[1];
    REAL const z_xy = 0.5*a / sqrt(argxy);
    REAL const g_xy = 2.0*a*erf(z_xy) + 4.0/sqrt_pi*sqrt(argxy)*(exp(-pow(z_xy,2))-1.0);

    REAL const pa_pD = -a*a/sqrt_pi * exp(-1.0*pow(z_xy,2))*pow(argxy,-3.0/2.0) * x;
    REAL const pb_pD = 4.0/sqrt_pi * pow(argxy,-0.5) * x * ((0.5+pow(z_xy,2))*exp(-1.0*pow(z_xy,2))-0.5);
    REAL const pgxy_pD = pa_pD + pb_pD;

    value[point_index] = 1.0/prefix * g_xy*g_xy * pow(argz,-0.5) + p[2];

    // derivatives
    REAL * current_derivatives = derivative + point_index;
    // D
    current_derivatives[0 * n_points] = 1.0/prefix*2.0*g_xy*pgxy_pD*pow(argz,-0.5) + 1.0/prefix*g_xy*g_xy*(-0.5)*pow(argz,-3.0/2.0)*x/pow(sigma_z,2);
    // N
    current_derivatives[1 * n_points] = -1.0/(4.0*a*a*sqrt_pi) * pow(p[1],-2)*pow(g_xy,2.0)*pow(argz,-0.5);
    // G_inf
    current_derivatives[2 * n_points] = 1.0;
}

#endif