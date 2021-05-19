#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <array>

double _2d_gaussian_function(int x, int y, double std_dev) {
    double std_dev_2 = 2 * std_dev * std_dev;
    int x_2 = x * x;
    int y_2 = y * y;

    double e_power = -1.0 * (double)(x_2+y_2) / std_dev_2;
    double e = exp(e_power);
    double co = 1.0 / (M_PI * std_dev_2);

    return co * e;
}

double _1d_gaussian_function(int x, double std_dev) {
    double std_dev_2 = 2 * std_dev * std_dev;
    int x_2 = x * x;

    double e_power = -1.0 * (double)x_2 / std_dev_2;
    double e = exp(e_power);
    double co = 1.0 / sqrt(M_PI * std_dev_2);

    return co * e;
}

double* _1d_blur_kernel(int kernel_size, double std_dev) {

    double* kernel = new double[kernel_size];

    double sum = 0;
    int k = kernel_size / 2;

    double* blur = new double[k + 1];

    for (int i = 0; i < k + 1; ++i)
        blur[i] = _1d_gaussian_function(i, std_dev);

    for (int i = 0; i < kernel_size; ++i) {
        int x = abs(i - k);
        kernel[i] = blur[x];
        sum += kernel[i];
    }

    // as we can not cover 100% of the whole normal distribution we need to renormalize the kernel so that its sum is 1 again
    // if we dont do this, the pixel would become slightly brighter, especially if the kernel dimension is to small
    double renormalize = 1.0 / sum;
    for (int i = 0; i < kernel_size; ++i)
        kernel[i] = renormalize * kernel[i];

    delete[] blur;

    return kernel;
}

double* _2d_blur_kernel(int kernel_size, double std_dev) {
    double** kernel = new double* [kernel_size];
    for (int i = 0; i < kernel_size; i++)
        kernel[i] = new double[kernel_size];

    double sum = 0;
    int k = kernel_size / 2;

    double** blur = new double * [k + 1];
    for (int i = 0; i < k + 1; i++)
        blur[i] = new double[k + 1];

    for (int i = 0; i < k+1; ++i) {
        for (int j = 0; j < k+1; ++j) {
            blur[i][j] = _2d_gaussian_function(i, j, std_dev);
        }
    }

    for (int i = 0; i < kernel_size; ++i) {
        int x = abs(i - k);
        for (int j = 0; j < kernel_size; ++j) {
            int y = abs(j - k);
            kernel[i][j] = blur[x][y];
            sum += kernel[i][j];
        }
    }

    double * simpleKernel = new double[kernel_size*kernel_size];

    // as we can not cover 100% of the whole normal distribution we need to renormalize the kernel so that its sum is 1 again
    // if we dont do this, the pixel would become slightly brighter, especially if the kernel dimension is to small
    double renormalize = 1.0 / sum;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int index = i * kernel_size + j;
            simpleKernel[index] = renormalize * kernel[i][j];
        }
    }

    for (int i = 0; i < k + 1; i++)
        delete[] blur[i];

    delete[] blur;

    for (int i = 0; i < kernel_size; i++)
        delete[] kernel[i];

    delete[] kernel;

    return simpleKernel;
}

