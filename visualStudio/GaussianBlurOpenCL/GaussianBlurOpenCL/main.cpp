#include <iostream>
#include <fstream>
#include <cmath>
#include "cxxopts.hpp"
#include "gaussian_blur.h"
#include "tga.h"
#include <memory>
#include "CL/cl.h"

std::string cl_errorstring(cl_int err);
void checkStatus(cl_int err);
void printCompilerError(cl_program program, cl_device_id device);

void blur_pixel(
    const unsigned char* r,
    const unsigned char* g,
    const unsigned char* b,
    unsigned char* rOut,
    unsigned char* gOut,
    unsigned char* bOut,
    int width,
    int height,
    int px,
    int py,
    int kernelSize,
    const double* kernel
) {

    double rBlur = 0.0;
    double gBlur = 0.0;
    double bBlur = 0.0;

    int i = 0;
    for (int x = px - (kernelSize / 2); x <= px + (kernelSize / 2); x++) {
        int j = 0;
        for (int y = py - (kernelSize / 2); y <= py + (kernelSize / 2); y++) {
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

int main(int argc, char** argv) {
    cxxopts::Options options("Gaussian Blur", "This program can be used to apply gaussian blur to an image");
    options.add_options()
        ("i,inFilePath", "Path to the image file to blur", cxxopts::value<std::string>())
        ("o,outFilePath", "Where the blurred image should be written to", cxxopts::value<std::string>())
        ("k,kernelSize", "Size of the kernel", cxxopts::value<int>())
        ("s,sigma", "Sigma to use for the kernel calculation", cxxopts::value<double>());

    auto result = options.parse(argc, argv);

    struct BlurOptions blurOptions;
    blurOptions.inFilePath = result["inFilePath"].as<std::string>();
    blurOptions.outFilePath = result["outFilePath"].as<std::string>();
    blurOptions.kernelSize = result["kernelSize"].as<int>();
    blurOptions.sigma = result["sigma"].as<double>();

    int size = blurOptions.kernelSize;
    double std_dev = blurOptions.sigma;

    if (size <= 0 || size > 9 || size % 2 == 0) {
        std::cout << "invalid kernel size" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (std_dev <= 0) {
        std::cout << "invalid sigma" << std::endl;
        exit(EXIT_FAILURE);
    }

    // used for checking error status of api calls
    cl_int status;

    // retrieve the number of platforms
    cl_uint numPlatforms = 0;
    checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

    if (numPlatforms == 0)
    {
        printf("Error: No OpenCL platform available!\n");
        exit(EXIT_FAILURE);
    }

    // select the platform
    cl_platform_id platform;
    checkStatus(clGetPlatformIDs(1, &platform, NULL));

    // retrieve the number of devices
    cl_uint numDevices = 0;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

    if (numDevices == 0)
    {
        printf("Error: No OpenCL device available for platform!\n");
        exit(EXIT_FAILURE);
    }

    // select the device
    cl_device_id device;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

    // create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkStatus(status);

    // create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
    checkStatus(status);

    double* blur = blur_kernel(size, std_dev);

    tga::TGAImage image;
    tga::LoadTGA(&image, blurOptions.inFilePath.c_str());

    auto r = std::make_unique<unsigned char[]>(image.height * image.width);
    auto g = std::make_unique<unsigned char[]>(image.height * image.width);
    auto b = std::make_unique<unsigned char[]>(image.height * image.width);

    for (int i = 0; i < image.height * image.width; i++) {
        r[i] = image.imageData[i * 3 + 0];
        g[i] = image.imageData[i * 3 + 1];
        b[i] = image.imageData[i * 3 + 2];
    }

    auto rOut = std::make_unique<unsigned char[]>(image.height * image.width);
    auto gOut = std::make_unique<unsigned char[]>(image.height * image.width);
    auto bOut = std::make_unique<unsigned char[]>(image.height * image.width);

    auto dataSize = sizeof(unsigned char) * image.height * image.width;
    cl_mem bufferR = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferG = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferROut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferGOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferBOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
    checkStatus(status);

    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferR, CL_TRUE, 0, dataSize, r.get(), 0, NULL, NULL));
    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferG, CL_TRUE, 0, dataSize, g.get(), 0, NULL, NULL));
    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0, dataSize, b.get(), 0, NULL, NULL));

    // read the kernel source
    const char* kernelFileName = "gauss.cl";
    std::ifstream ifs(kernelFileName);
    if (!ifs.good())
    {
        printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
        exit(EXIT_FAILURE);
    }

    std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    const char* programSourceArray = programSource.c_str();
    size_t programSize = programSource.length();

    // create the program
    cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char**>(&programSourceArray), &programSize, &status);
    checkStatus(status);

    // build the program
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        printCompilerError(program, device);
        exit(EXIT_FAILURE);
    }

    // create the vector addition kernel
    cl_kernel kernel = clCreateKernel(program, "test", &status);
    checkStatus(status);

    checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferR));
    checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferG));
    checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferB));
    checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferROut));
    checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferGOut));
    checkStatus(clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufferBOut));

    size_t globalWorkSize = image.height * image.width;
    checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL));

    checkStatus(clEnqueueReadBuffer(commandQueue, bufferROut, CL_TRUE, 0, dataSize, rOut.get(), 0, NULL, NULL));
    checkStatus(clEnqueueReadBuffer(commandQueue, bufferGOut, CL_TRUE, 0, dataSize, gOut.get(), 0, NULL, NULL));
    checkStatus(clEnqueueReadBuffer(commandQueue, bufferBOut, CL_TRUE, 0, dataSize, bOut.get(), 0, NULL, NULL));

    checkStatus(clReleaseKernel(kernel));
    checkStatus(clReleaseProgram(program));
    checkStatus(clReleaseMemObject(bufferR));
    checkStatus(clReleaseCommandQueue(commandQueue));
    checkStatus(clReleaseContext(context));

    //for (int i = 0; i < image.height; i++) {
    //    for (int j = 0; j < image.width; j++) {
    //        blur_pixel(r.get(), g.get(), b.get(), rOut.get(), gOut.get(), bOut.get(), (int)image.width, (int)image.height, j, i, size, blur);
    //    }
    //}

    for (int i = 0; i < image.height * image.width; i++) {
        image.imageData[i * 3 + 0] = rOut[i];
        image.imageData[i * 3 + 1] = gOut[i];
        image.imageData[i * 3 + 2] = bOut[i];
    }

    tga::saveTGA(image, blurOptions.outFilePath.c_str());

    return 0;
}

std::string cl_errorstring(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:									return std::string("Success");
    case CL_DEVICE_NOT_FOUND:							return std::string("Device not found");
    case CL_DEVICE_NOT_AVAILABLE:						return std::string("Device not available");
    case CL_COMPILER_NOT_AVAILABLE:						return std::string("Compiler not available");
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return std::string("Memory object allocation failure");
    case CL_OUT_OF_RESOURCES:							return std::string("Out of resources");
    case CL_OUT_OF_HOST_MEMORY:							return std::string("Out of host memory");
    case CL_PROFILING_INFO_NOT_AVAILABLE:				return std::string("Profiling information not available");
    case CL_MEM_COPY_OVERLAP:							return std::string("Memory copy overlap");
    case CL_IMAGE_FORMAT_MISMATCH:						return std::string("Image format mismatch");
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return std::string("Image format not supported");
    case CL_BUILD_PROGRAM_FAILURE:						return std::string("Program build failure");
    case CL_MAP_FAILURE:								return std::string("Map failure");
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return std::string("Misaligned sub buffer offset");
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return std::string("Exec status error for events in wait list");
    case CL_INVALID_VALUE:                    			return std::string("Invalid value");
    case CL_INVALID_DEVICE_TYPE:              			return std::string("Invalid device type");
    case CL_INVALID_PLATFORM:                 			return std::string("Invalid platform");
    case CL_INVALID_DEVICE:                   			return std::string("Invalid device");
    case CL_INVALID_CONTEXT:                  			return std::string("Invalid context");
    case CL_INVALID_QUEUE_PROPERTIES:         			return std::string("Invalid queue properties");
    case CL_INVALID_COMMAND_QUEUE:            			return std::string("Invalid command queue");
    case CL_INVALID_HOST_PTR:                 			return std::string("Invalid host pointer");
    case CL_INVALID_MEM_OBJECT:               			return std::string("Invalid memory object");
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  			return std::string("Invalid image format descriptor");
    case CL_INVALID_IMAGE_SIZE:               			return std::string("Invalid image size");
    case CL_INVALID_SAMPLER:                  			return std::string("Invalid sampler");
    case CL_INVALID_BINARY:                   			return std::string("Invalid binary");
    case CL_INVALID_BUILD_OPTIONS:            			return std::string("Invalid build options");
    case CL_INVALID_PROGRAM:                  			return std::string("Invalid program");
    case CL_INVALID_PROGRAM_EXECUTABLE:       			return std::string("Invalid program executable");
    case CL_INVALID_KERNEL_NAME:              			return std::string("Invalid kernel name");
    case CL_INVALID_KERNEL_DEFINITION:        			return std::string("Invalid kernel definition");
    case CL_INVALID_KERNEL:                   			return std::string("Invalid kernel");
    case CL_INVALID_ARG_INDEX:                			return std::string("Invalid argument index");
    case CL_INVALID_ARG_VALUE:                			return std::string("Invalid argument value");
    case CL_INVALID_ARG_SIZE:                 			return std::string("Invalid argument size");
    case CL_INVALID_KERNEL_ARGS:             			return std::string("Invalid kernel arguments");
    case CL_INVALID_WORK_DIMENSION:          			return std::string("Invalid work dimension");
    case CL_INVALID_WORK_GROUP_SIZE:          			return std::string("Invalid work group size");
    case CL_INVALID_WORK_ITEM_SIZE:           			return std::string("Invalid work item size");
    case CL_INVALID_GLOBAL_OFFSET:            			return std::string("Invalid global offset");
    case CL_INVALID_EVENT_WAIT_LIST:          			return std::string("Invalid event wait list");
    case CL_INVALID_EVENT:                    			return std::string("Invalid event");
    case CL_INVALID_OPERATION:                			return std::string("Invalid operation");
    case CL_INVALID_GL_OBJECT:                			return std::string("Invalid OpenGL object");
    case CL_INVALID_BUFFER_SIZE:              			return std::string("Invalid buffer size");
    case CL_INVALID_MIP_LEVEL:                			return std::string("Invalid mip-map level");
    case CL_INVALID_GLOBAL_WORK_SIZE:         			return std::string("Invalid gloal work size");
    case CL_INVALID_PROPERTY:                 			return std::string("Invalid property");
    default:                                  			return std::string("Unknown error code");
    }
}

void checkStatus(cl_int err)
{
    if (err != CL_SUCCESS) {
        printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
        exit(EXIT_FAILURE);
    }
}

void printCompilerError(cl_program program, cl_device_id device)
{
    cl_int status;
    size_t logSize;
    char* log;

    // get log size
    status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    checkStatus(status);

    // allocate space for log
    log = static_cast<char*>(malloc(logSize));
    if (!log)
    {
        exit(EXIT_FAILURE);
    }

    // read the log
    status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
    checkStatus(status);

    // print the log
    printf("Build Error: %s\n", log);
}
