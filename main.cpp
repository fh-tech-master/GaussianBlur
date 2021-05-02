#include <iostream>
#include <cmath>
#include "cxxopts.hpp"
#include "gaussian_blur.h"
#include "tga.h"

void blur_pixel(
        const unsigned char * r,
        const unsigned char * g,
        const unsigned char * b,
        unsigned char * rOut,
        unsigned char * gOut,
        unsigned char * bOut,
        int width,
        int height,
        int px,
        int py,
        int kernelSize,
        const double * kernel
        ) {

    double rBlur = 0.0;
    double gBlur = 0.0;
    double bBlur = 0.0;

    int i = 0;
    for (int x = px - (kernelSize/2); x <= px + (kernelSize/2); x++) {
        int j = 0;
        for (int y = py - (kernelSize/2); y <= py + (kernelSize/2); y++) {
            int newX = x;
            int newY = y;
            if (newX < 0) newX = 0;
            if (newX >= width) newX = width - 1;
            if (newY < 0) newY = 0;
            if (newY >= height) newY = height - 1;

            int index = newY * width + newX;
            int blurIndex = i * kernelSize + j;

            rBlur += (double)r[index] * kernel[blurIndex];
            gBlur += (double)g[index] * kernel[blurIndex];
            bBlur += (double)b[index] * kernel[blurIndex];

            j++;
        }
        i++;
    }

    int index = py * width + px;

    rOut[index] = (unsigned char)round(rBlur);
    gOut[index] = (unsigned char)round(gBlur);
    bOut[index] = (unsigned char)round(bBlur);
}

struct BlurOptions {
    std::string inFilePath;
    std::string outFilePath;
    int kernelSize;
    double sigma;
};

int main(int argc, char **argv) {
    cxxopts::Options options("Gaussian Blur", "This program can be used to apply gaussian blur to an image");
    options.add_options()
            ("i,inFilePath", "Path to the image file to blur", cxxopts::value<std::string>())
            ("o,outFilePath", "Where the blurred image should be written to", cxxopts::value<std::string>())
            ("k,kernelSize", "Size of the kernel", cxxopts::value<int>())
            ("s,sigma", "Sigma to use for the kernel calculation", cxxopts::value<double>());

    auto result = options.parse(argc, argv);

    struct BlurOptions blurOptions = {
            .inFilePath = result["inFilePath"].as<std::string>(),
            .outFilePath = result["outFilePath"].as<std::string>(),
            .kernelSize = result["kernelSize"].as<int>(),
            .sigma = result["sigma"].as<double>()
    };

    int size = blurOptions.kernelSize;
    double std_dev = blurOptions.sigma;

    if (size <= 0 || size > 9 || size%2 == 0) {
        std::cout << "invalid kernel size" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (std_dev <= 0) {
        std::cout << "invalid sigma" << std::endl;
        exit(EXIT_FAILURE);
    }

    double * blur = blur_kernel(size, std_dev);

    tga::TGAImage image;
    tga::LoadTGA(&image, "lena.tga");

    auto * r = new unsigned char [image.height * image.width];
    auto * g = new unsigned char [image.height * image.width];
    auto * b = new unsigned char [image.height * image.width];

    for (int i = 0; i < image.height * image.width; i++) {
        r[i] = image.imageData[i*3+0];
        g[i] = image.imageData[i*3+1];
        b[i] = image.imageData[i*3+2];
    }

    auto * rOut = new unsigned char [image.height * image.width];
    auto * gOut = new unsigned char [image.height * image.width];
    auto * bOut = new unsigned char [image.height * image.width];
    
    for (int i = 0; i < image.height; i++) {
        for (int j = 0; j < image.width; j++) {
            blur_pixel(r, g, b, rOut, gOut, bOut, (int)image.width, (int)image.height, j, i, size, blur);
        }
    }

    for (int i = 0; i < image.height * image.width; i++) {
        image.imageData[i*3+0] = rOut[i];
        image.imageData[i*3+1] = gOut[i];
        image.imageData[i*3+2] = bOut[i];
    }

    tga::saveTGA(image, "out.tga");

    return 0;
}
