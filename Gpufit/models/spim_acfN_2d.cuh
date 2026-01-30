#ifndef GPUFIT_SPIMACFN2D_CUH_INCLUDED
#define GPUFIT_SPIMACFN2D_CUH_INCLUDED

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
*             p[3]: sigma_xy
*             p[4]: sigma_z
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

__device__ void calculate_spim_acfN_2d(
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
    REAL const a = 0.145; // side length of square pixel in object space, in um
    //REAL const sigma_xy = 1.0; // radius of psf in xy plane
    //REAL const sigma_z = 1.0; // radius of psf in z plane

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
    REAL const sigma_xy = p[3];
    REAL const argxy = 4.0 * p[0] * x + pow(sigma_xy, 2);
    
    REAL const prefix = sqrt_pi * p[1];
    REAL const z_xy = a / sqrt(argxy);
    REAL const tmp1 = exp(-pow(z_xy, 2)) - 1.0;
    REAL const g_xy = erf(z_xy) + sqrt(argxy) / a / sqrt_pi * tmp1;


    value[point_index] = 1.0/prefix * g_xy*g_xy + p[2]; // this the total correlation function

    // derivatives
    REAL * current_derivatives = derivative + point_index;
    // D
    current_derivatives[0 * n_points] = 4.0 / a / p[1] / pi / sqrt(argxy) * x * tmp1 * g_xy;  
    // N
    current_derivatives[1 * n_points] = -1.0 / sqrt_pi * pow(p[1], -2) * pow(g_xy, 2.0);
    // G_inf
    current_derivatives[2 * n_points] = 1.0;
    // sigma_xy
    current_derivatives[3 * n_points] = 2.0 * tmp1 * p[3] * g_xy / (a * p[1] * pi * pow(argxy, 0.5));
}

#endif