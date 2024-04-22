#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
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
#include <cstdint>
#include "KernelArguments.hpp"
#include "Math.hpp"
#include "BufferUtils.hpp"
#include "hipblaslt_float8.h"

template<typename T>
void dumpBuffer(const char* name, const std::vector<T>& data)
{
    std::cout << "------ " << name << " ------" << std::endl;
    for (int i=0; i<data.size(); i++)
    {
        std::cout << float(data[i]);
        if (i!=0 && (i%128 == 0))
            std::cout << std::endl;
        else
            std::cout << " ";
    }
    std::cout << std::endl;
}


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

template <typename Ti, typename To, typename Ts>
void cpuAMaxWithScale(To* out, Ts* outD, Ti* in, const float* in_scale, std::uint32_t length)
{
    // calculate amax
    Ti m = 0;
    for(int j = 0; j < length; j++)
    {
        m       = max(m, abs(in[j]));
        outD[j] = static_cast<Ts>(in[j] * in_scale[0]);
    }
    out[0] = To(m);
}

template<typename Ti, typename To>
hipError_t launchASMAMax(hipFunction_t func, To *out, Ti* in, Ti *wk, std::uint32_t* sy, std::uint32_t length, std::uint32_t workSize, std::uint32_t numGroups, std::uint32_t numRuns) {

    std::uint32_t workgroups = min((length + workSize - 1) / workSize, numGroups);
    std::cout << "workgroups " << workgroups << std::endl;

    KernelArguments args;
    args.append(out);
    args.append(in);
    args.append(wk);
    args.append(sy);
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

template<typename Ti, typename To, typename Ts>
hipError_t launchASMAMaxScale(hipFunction_t func, To *out, Ts* outD, Ti* in, float* scale, Ti *wk, std::uint32_t* sy, std::uint32_t length, std::uint32_t workSize, std::uint32_t numGroups, std::uint32_t numRuns) {

    std::uint32_t workgroups = min((length + workSize - 1) / workSize, numGroups);
    std::cout << "workgroups " << workgroups << std::endl;

    KernelArguments args;
    args.append(out);
    args.append(outD); // scale result
    args.append(in);
    args.append(scale); // scale input
    args.append(wk);
    args.append(sy);
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

    Ti *workspace{};
    err = hipMalloc(&workspace, sizeof(Ti) * numGroups);
    err = hipMemset(workspace, 0, sizeof(Ti) * numGroups);

    std::uint32_t *sync{};
    err = hipMalloc(&sync, sizeof(std::uint32_t));
    err = hipMemset(sync, 0, sizeof(std::uint32_t));

    hipModule_t module{};
    hipFunction_t func{};
    int numRun = 1;  //(1000000.0f * 16 * 1024 / float(length));
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

    To error = 0.0;
    for (std::size_t i = 0; i < 1; ++i) {
        error = max(error, abs(cpuOutput[i]-cpuRef[i]));
    }
    std::cout << "Tony CPU " << cpuRef[0] << " GPU " << cpuOutput[0] << " max error : " << float(error) << std::endl;

    err = hipFree(gpuOutput);
    err = hipFree(gpuInput);
    err = hipFree(workspace);
    err = hipModuleUnload(module);
}


template <typename Ti, typename To, typename Ts>
void AMaxScaleTest(const std::string& coPath, const std::uint32_t& length, const std::uint32_t& workSize, const std::uint32_t& numGroups)
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

    Ti *workspace{};
    err = hipMalloc(&workspace, sizeof(Ti) * numGroups);
    err = hipMemset(workspace, 0, sizeof(Ti) * numGroups);

    std::uint32_t *sync{};
    err = hipMalloc(&sync, sizeof(std::uint32_t));
    err = hipMemset(sync, 0, sizeof(std::uint32_t));

    float scale = 0.5;
    float *gpuScale{};
    err = hipMalloc(&gpuScale, sizeof(float));
    err = hipMemcpyHtoD(gpuScale, &scale, sizeof(float));

    Ts *gpuOutputD{};
    err = hipMalloc(&gpuOutputD, sizeof(Ts) * length);
    err = hipMemset(gpuOutputD, 0, sizeof(Ts) * length);

    hipModule_t module{};
    hipFunction_t func{};
    int numRun = 1; //(1000000.0f * 16 * 1024 / float(length));
    std::cout << "numRun " << numRun << std::endl;

    err = prepareASMKernel("AMax_Scale", coPath, &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;

    err = launchASMAMaxScale(func, gpuOutput, gpuOutputD, gpuInput, gpuScale, workspace, sync, length, workSize, numGroups, numRun);
    if (err)
        std::cout << "launchASMAMax error : " << err << std::endl;

    std::vector<To> cpuOutput(1, 0.0f);
    err = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(To));
    // dumpBuffer("GPU result", cpuOutput, cpuOutput.size());

    std::vector<Ts> cpuOutD(length);
    err = hipMemcpyDtoH(cpuOutD.data(), gpuOutputD, length * sizeof(Ts));
    // dumpBuffer("cpuOutD", cpuOutD);

    std::vector<To> cpuWs(numGroups, 0.f);
    err = hipMemcpyDtoH(cpuWs.data(), workspace, numGroups * sizeof(To));
    // dumpBuffer("WS result", cpuWs, cpuWs.size());

    std::vector<uint32_t> cpusy(1, 0);
    err = hipMemcpyDtoH(cpusy.data(), sync, sizeof(uint32_t));
    // dumpBuffer("sy result", cpusy, cpusy.size());

    std::vector<To> cpuRef(1, 0.f);
    std::vector<Ts> cpuRefOutD(length);
    cpuAMaxWithScale<Ti, To, Ts>(cpuRef.data(), cpuRefOutD.data(), cpuInput.data(), &scale, length);
    // dumpBuffer("cpuRefOutD", cpuRefOutD);

    To error = 0.0;
    for (std::size_t i = 0; i < 1; ++i) {
        error = max(error, abs(cpuOutput[i]-cpuRef[i]));
    }

    std::cout << "Tony CPU " << cpuRef[0] << " GPU " << cpuOutput[0] << " max error : " << float(error) << std::endl;

    error = 0.0;
    for (std::size_t i = 0; i < length; i++) {
        error = max(error, abs(float(cpuOutD[i])-float(cpuRefOutD[i])));
    }

    std::cout << "Tony CPU  GPU OutD max error : " << float(error) << std::endl;

    err = hipFree(gpuOutput);
    err = hipFree(gpuInput);
    err = hipFree(workspace);
    err = hipFree(sync);
    err = hipFree(gpuScale);
    err = hipFree(gpuOutputD);
    err = hipModuleUnload(module);
}


int main(int argc, char **argv) {

    if (argc != 4)
    {
        std::cout << "amax [W] [H] [scale]" << std::endl;
        return -1;
    }

    const std::uint32_t m(std::atoi(argv[1]));
    const std::uint32_t n(std::atoi(argv[2]));
    const std::uint32_t is_scale(std::atoi(argv[3]));

    const std::uint32_t length = m * n;


    std::cout << " m " << m << " n " << n << std::endl;

    if (is_scale)
        AMaxScaleTest<float, float, hipblaslt_f8_fnuz>("amax-scale.co", length, 131072, 128);
    else
        AMaxTest<_Float16, float>("amax.co", length, 131072, 128);

    return 0;
}
