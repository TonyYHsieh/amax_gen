#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>
#include <hip/math_functions.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include "KernelArguments.hpp"
#include "Math.hpp"
#include "BufferUtils.hpp"


template<typename T>
T abs(T a)
{
    return (a > T(0)) ? a : -a;
}


template<typename T>
T max(T a, T b)
{
    return (a > b) ? a : b;
}

template<typename T>
T min(T a, T b)
{
    return (a > b) ? b : a;
}

template<typename Ti, typename To>
void cpuAMax(To *out, Ti *in, std::uint32_t length)
{
    // calculate amax
    out[0] = To(0);
    for(int j=0; j<length; j++) {
        out[0] = max(out[0], To(abs(in[j])));
    }
}

template<typename Ti, typename To>
hipError_t launchASMAMax(hipFunction_t func, To *out, Ti* in, To *wk, std::uint32_t* sync,  std::uint32_t length, std::uint32_t workSize, std::uint32_t numGroups, std::uint32_t numRuns) {

    std::uint32_t workgroups = min((length + workSize - 1) / workSize, numGroups);
//    std::uint32_t workgroups = (length + workSize -1) / workSize;
    std::cout << "workgroups " << workgroups << std::endl;

    KernelArguments args;
    args.append(out);
    args.append(in);
    args.append(wk);
    args.append(sync);
    args.append(length);
    args.append(workSize);
    args.append(workgroups);
    args.applyAlignment();
    std::size_t argsSize = args.size();
    void *launchArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        args.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argsSize,
        HIP_LAUNCH_PARAM_END
    };

    hipEvent_t beg, end;
    auto err = hipEventCreate(&beg);
    err = hipEventCreate(&end);

    for (size_t i = 0; i < numRuns; ++i) {
        err = hipExtModuleLaunchKernel(func, 256 * workgroups, 1, 1, 256, 1, 1, 1000 * sizeof(float), nullptr, nullptr, launchArgs);
    }

    err = hipEventRecord(beg);
    for (size_t i = 0; i < numRuns; ++i) {
        err = hipExtModuleLaunchKernel(func, 256 * workgroups, 1, 1, 256, 1, 1, 1000 * sizeof(float), nullptr, nullptr, launchArgs);
    }
    err = hipEventRecord(end);
    err = hipEventSynchronize(end);
    err = hipDeviceSynchronize();

    float dur{};
    err = hipEventElapsedTime(&dur, beg, end);
    std::cout << "ASM kernel time: " << std::to_string(dur / numRuns * 1000) << " us\n";
    return err;
}

hipError_t prepareASMKernel(const std::string &funcName, const std::string &coPath, hipModule_t *module, hipFunction_t *func) {
    auto err = hipModuleLoad(module, coPath.c_str());
    if (err != hipSuccess)
        std::cout << "hipModuleLoad failed" << std::endl;
    err = hipModuleGetFunction(func, *module, funcName.c_str());
    if (err != hipSuccess)
        std::cout << "hipModuleGetFunction failed" << std::endl;
    return err;
}

template <typename T>
void dumpBuffer(const char* title, const std::vector<T>& data, int length)
{
    std::cout << "----- " << title << " -----" << std::endl;
    for(int j=0; j<length; j++)
    {
        std::cout << float(data[j]) << " ";
    }
    std::cout << std::endl;
}

template <typename Ti, typename To>
void AMaxTest(const std::string& coPath, const std::uint32_t& length, const std::uint32_t& workSize, const std::uint32_t& numGroups)
{
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);

    std::vector<Ti> cpuInput(length, 0);
    randomize(begin(cpuInput), end(cpuInput));

    Ti *gpuInput{};
    err = hipMalloc(&gpuInput, sizeof(Ti) * length);
    err = hipMemcpyHtoD(gpuInput, cpuInput.data(), cpuInput.size() * sizeof(Ti));

    To *gpuOutput{};
    err = hipMalloc(&gpuOutput, sizeof(To));
    err = hipMemset(gpuOutput, 0, sizeof(To));

    To *workspace{};
    err = hipMalloc(&workspace, sizeof(To) * numGroups);
    err = hipMemset(workspace, 0, sizeof(To) * numGroups);

    std::uint32_t *sync{};
    err = hipMalloc(&sync, sizeof(std::uint32_t));
    err = hipMemset(sync, 0, sizeof(std::uint32_t));

    hipModule_t module{};
    hipFunction_t func{};
    int numRun = (1000000.0f * 16 * 1024 / float(length));
    std::cout << "numRun " << numRun << std::endl;

    err = prepareASMKernel("AMax", coPath, &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;

    err = launchASMAMax(func, gpuOutput, gpuInput, workspace, sync, length, workSize, numGroups, numRun);
    if (err)
        std::cout << "launchASMAMax error : " << err << std::endl;

    std::vector<To> cpuOutput(1, 0.0f);
    err = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(To));
    // dumpBuffer("GPU result", cpuOutput, cpuOutput.size());

    std::vector<To> cpuRef(1, 0.f);
    cpuAMax<Ti, To>(cpuRef.data(), cpuInput.data(), length);
    // dumpBuffer("CPU result", cpuRef, cpuRef.size());

    std::vector<To> cpuWs(numGroups, 0.f);
    err = hipMemcpyDtoH(cpuWs.data(), workspace, numGroups * sizeof(To));
    // dumpBuffer("WS result", cpuWs, cpuWs.size());

    std::vector<uint32_t> cpuSy(1, 0);
    err = hipMemcpyDtoH(cpuSy.data(), sync, sizeof(uint32_t));
    // dumpBuffer("Sy result", cpuSy, cpuSy.size());


    To error = 0.0;
    for (std::size_t i = 0; i < 1; ++i) {
        error = max(error, abs(cpuOutput[i]-cpuRef[i]));
    }

    std::cout << "Tony CPU " << cpuRef[0] << " GPU " << cpuOutput[0] << " max error : " << float(error) << std::endl;

    err = hipFree(gpuOutput);
    err = hipFree(gpuInput);
    err = hipModuleUnload(module);
}


int main(int argc, char **argv) {

    if (argc != 8)
    {
        std::cout << "amax amax.co H S [W] [H] [blockSize] [numGroups]" << std::endl;
        return -1;
    }

    const std::string coPath(argv[1]);
    const std::string inType(argv[2]);
    const std::string outType(argv[3]);
    const std::uint32_t m(std::atoi(argv[4]));
    const std::uint32_t n(std::atoi(argv[5]));
    const std::uint32_t workSize(std::atoi(argv[6]));
    const std::uint32_t numGroups(std::atoi(argv[7]));
    const std::uint32_t length = m * n;

    std::cout << "inType " << inType << " outType " << outType << " m " << m << " n " << n << " workSize " << workSize << " numGroups " << numGroups << std::endl;

    if (inType == "H" and outType == "S")
        AMaxTest<_Float16, float>(coPath, length, workSize, numGroups);
    else
        std::cout << "unsupported type " << inType << " " << outType << std::endl;

    return 0;
}
