#include "bridge.h"
#include "../metal_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <Availability.h>
#include <stdlib.h>
#include <string.h>

@interface MFPlan : NSObject {
@public
    int32_t transform;
    BOOL oddLastDimension;
    size_t realBytes;
    size_t HermitianBytes;
    NSArray<NSNumber *> *realShape;
    NSArray<NSNumber *> *HermitianShape;
    NSArray<NSNumber *> *axes;
    MPSGraph *forwardGraph;
    MPSGraphTensor *forwardInput;
    MPSGraphTensor *forwardOutput;
    MPSGraph *inverseGraph;
    MPSGraphTensor *inverseInput;
    MPSGraphTensor *inverseOutput;
}
@end

@implementation MFPlan
@end

static void mf_set_error(char **destination, NSString *message) {
    if (destination == nullptr) {
        return;
    }
    const char *text = message == nil ? "unknown Metal FFT failure" : message.UTF8String;
    *destination = strdup(text == nullptr ? "unknown Metal FFT failure" : text);
}

static void mf_copy_runtime_error(char **destination, char *runtimeError) {
    if (runtimeError == nullptr) {
        mf_set_error(destination, @"unknown Metal runtime failure");
        return;
    }
    if (destination != nullptr) {
        *destination = strdup(runtimeError);
    }
    mr_free_error(runtimeError);
}

static NSArray<NSNumber *> *mf_shape(const int64_t *dimensions,
                                     size_t rank,
                                     int64_t batch,
                                     BOOL Hermitian) {
    NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:rank + (batch > 1 ? 1 : 0)];
    if (batch > 1) {
        [shape addObject:@(batch)];
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        int64_t value = dimensions[axis];
        if (Hermitian && axis + 1 == rank) {
            value = value / 2 + 1;
        }
        [shape addObject:@(value)];
    }
    return [shape copy];
}

static NSArray<NSNumber *> *mf_axes(size_t rank, int64_t batch) {
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:rank];
    NSInteger offset = batch > 1 ? 1 : 0;
    for (size_t axis = 0; axis < rank; ++axis) {
        [result addObject:@(offset + (NSInteger)axis)];
    }
    return [result copy];
}

static size_t mf_element_count(NSArray<NSNumber *> *shape) {
    size_t result = 1;
    for (NSNumber *dimension in shape) {
        result *= dimension.unsignedLongLongValue;
    }
    return result;
}

static void mf_build_forward_graph(MFPlan *plan) {
    plan->forwardGraph = [MPSGraph new];
    if (plan->transform == MF_R2C) {
        plan->forwardInput = [plan->forwardGraph placeholderWithShape:plan->realShape
                                                             dataType:MPSDataTypeFloat32
                                                                 name:@"mumax3_fft_r2c_input"];
        MPSGraphFFTDescriptor *descriptor = [MPSGraphFFTDescriptor descriptor];
        descriptor.inverse = NO;
        descriptor.scalingMode = MPSGraphFFTScalingModeNone;
        descriptor.roundToOddHermitean = plan->oddLastDimension;
        plan->forwardOutput = [plan->forwardGraph realToHermiteanFFTWithTensor:plan->forwardInput
                                                                          axes:plan->axes
                                                                    descriptor:descriptor
                                                                          name:@"mumax3_fft_r2c_output"];
    } else {
        plan->forwardInput = [plan->forwardGraph placeholderWithShape:plan->realShape
                                                             dataType:MPSDataTypeComplexFloat32
                                                                 name:@"mumax3_fft_c2c_forward_input"];
        MPSGraphFFTDescriptor *descriptor = [MPSGraphFFTDescriptor descriptor];
        descriptor.inverse = NO;
        descriptor.scalingMode = MPSGraphFFTScalingModeNone;
        plan->forwardOutput = [plan->forwardGraph fastFourierTransformWithTensor:plan->forwardInput
                                                                            axes:plan->axes
                                                                      descriptor:descriptor
                                                                            name:@"mumax3_fft_c2c_forward_output"];
    }
}

static void mf_build_inverse_graph(MFPlan *plan) {
    plan->inverseGraph = [MPSGraph new];
    MPSGraphFFTDescriptor *descriptor = [MPSGraphFFTDescriptor descriptor];
    descriptor.inverse = YES;
    descriptor.scalingMode = MPSGraphFFTScalingModeNone;
    descriptor.roundToOddHermitean = plan->oddLastDimension;

    if (plan->transform == MF_C2R) {
        plan->inverseInput = [plan->inverseGraph placeholderWithShape:plan->HermitianShape
                                                             dataType:MPSDataTypeComplexFloat32
                                                                 name:@"mumax3_fft_c2r_input"];
        plan->inverseOutput = [plan->inverseGraph HermiteanToRealFFTWithTensor:plan->inverseInput
                                                                          axes:plan->axes
                                                                    descriptor:descriptor
                                                                          name:@"mumax3_fft_c2r_output"];
    } else {
        plan->inverseInput = [plan->inverseGraph placeholderWithShape:plan->realShape
                                                             dataType:MPSDataTypeComplexFloat32
                                                                 name:@"mumax3_fft_c2c_inverse_input"];
        plan->inverseOutput = [plan->inverseGraph fastFourierTransformWithTensor:plan->inverseInput
                                                                            axes:plan->axes
                                                                      descriptor:descriptor
                                                                            name:@"mumax3_fft_c2c_inverse_output"];
    }
}

static MPSGraphTensorData *mf_tensor_data(id<MTLBuffer> buffer,
                                          size_t offset,
                                          NSArray<NSNumber *> *shape,
                                          MPSDataType dataType) {
    if (offset == 0) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                       shape:shape
                                                    dataType:dataType];
    }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        MPSNDArrayDescriptor *descriptor = [MPSNDArrayDescriptor descriptorWithDataType:dataType
                                                                                  shape:shape];
        descriptor.preferPackedRows = YES;
        MPSNDArray *array = [[MPSNDArray alloc] initWithBuffer:buffer
                                                       offset:offset
                                                   descriptor:descriptor];
        return [[MPSGraphTensorData alloc] initWithMPSNDArray:array];
    }
#endif
    return nil;
}

extern "C" void *mf_plan_create(const int64_t *dimensions,
                                  size_t rank,
                                  int64_t batch,
                                  int32_t transform,
                                  char **error_message) {
    @autoreleasepool {
        if (dimensions == nullptr || rank == 0 || rank > 3 || batch < 1) {
            mf_set_error(error_message, @"rank must be 1...3 and batch must be positive");
            return nullptr;
        }
        if (transform != MF_R2C && transform != MF_C2R && transform != MF_C2C) {
            mf_set_error(error_message, @"only single-precision R2C, C2R, and C2C transforms are supported");
            return nullptr;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            if (dimensions[axis] < 1) {
                mf_set_error(error_message, @"all FFT dimensions must be positive");
                return nullptr;
            }
        }
        if (@available(macOS 14.0, *)) {
            @try {
                MFPlan *plan = [MFPlan new];
                plan->transform = transform;
                plan->oddLastDimension = dimensions[rank - 1] % 2 != 0;
                plan->realShape = mf_shape(dimensions, rank, batch, NO);
                plan->HermitianShape = mf_shape(dimensions, rank, batch, YES);
                plan->axes = mf_axes(rank, batch);
                plan->realBytes = mf_element_count(plan->realShape) * sizeof(float);
                plan->HermitianBytes = mf_element_count(plan->HermitianShape) * 2 * sizeof(float);

                if (transform == MF_R2C || transform == MF_C2C) {
                    mf_build_forward_graph(plan);
                }
                if (transform == MF_C2R || transform == MF_C2C) {
                    mf_build_inverse_graph(plan);
                }
                return (__bridge_retained void *)plan;
            } @catch (NSException *exception) {
                mf_set_error(error_message, exception.reason);
                return nullptr;
            }
        }
        mf_set_error(error_message, @"MPSGraph FFT requires macOS 14 or newer");
        return nullptr;
    }
}

extern "C" int mf_plan_execute(void *opaquePlan,
                                 const void *input,
                                 void *output,
                                 int32_t direction,
                                 char **error_message) {
    @autoreleasepool {
        if (opaquePlan == nullptr || input == nullptr || output == nullptr) {
            mf_set_error(error_message, @"plan and buffers must be non-null");
            return MF_ERROR_INVALID_ARGUMENT;
        }
        if (@available(macOS 14.0, *)) {
            // Continue below. Keeping the availability check here produces a
            // clear runtime error when an otherwise relocatable binary is
            // started on an older macOS release.
        } else {
            mf_set_error(error_message, @"MPSGraph FFT requires macOS 14 or newer");
            return MF_ERROR_UNAVAILABLE;
        }

        MFPlan *plan = (__bridge MFPlan *)opaquePlan;
        const BOOL inverse = plan->transform == MF_C2R ||
                             (plan->transform == MF_C2C && direction > 0);
        const size_t inputBytes = plan->transform == MF_R2C
                                      ? plan->realBytes
                                      : (plan->transform == MF_C2R ? plan->HermitianBytes : plan->realBytes * 2);
        const size_t outputBytes = plan->transform == MF_R2C
                                       ? plan->HermitianBytes
                                       : (plan->transform == MF_C2R ? plan->realBytes : plan->realBytes * 2);

        mr_external_context context = {};
        char *runtimeError = nullptr;
        int runtimeStatus = mr_begin_external(&context, &runtimeError);
        if (runtimeStatus != MR_SUCCESS) {
            mf_copy_runtime_error(error_message, runtimeError);
            return MF_ERROR_RUNTIME;
        }

        int result = MF_SUCCESS;
        @try {
            mr_buffer_view inputView = {};
            runtimeStatus = mr_resolve_buffer_locked(input, inputBytes, &inputView, &runtimeError);
            if (runtimeStatus != MR_SUCCESS) {
                mf_copy_runtime_error(error_message, runtimeError);
                result = MF_ERROR_RUNTIME;
            }

            mr_buffer_view outputView = {};
            if (result == MF_SUCCESS) {
                runtimeStatus = mr_resolve_buffer_locked(output, outputBytes, &outputView, &runtimeError);
                if (runtimeStatus != MR_SUCCESS) {
                    mf_copy_runtime_error(error_message, runtimeError);
                    result = MF_ERROR_RUNTIME;
                }
            }

            if (result == MF_SUCCESS) {
                MPSGraph *graph = inverse ? plan->inverseGraph : plan->forwardGraph;
                MPSGraphTensor *inputTensor = inverse ? plan->inverseInput : plan->forwardInput;
                MPSGraphTensor *outputTensor = inverse ? plan->inverseOutput : plan->forwardOutput;
                NSArray<NSNumber *> *inputShape =
                    plan->transform == MF_R2C ? plan->realShape :
                    (plan->transform == MF_C2R ? plan->HermitianShape : plan->realShape);
                NSArray<NSNumber *> *outputShape =
                    plan->transform == MF_R2C ? plan->HermitianShape :
                    (plan->transform == MF_C2R ? plan->realShape : plan->realShape);
                MPSDataType dataType =
                    plan->transform == MF_R2C ? MPSDataTypeFloat32 :
                    (plan->transform == MF_C2R ? MPSDataTypeComplexFloat32 : MPSDataTypeComplexFloat32);
                MPSDataType outputDataType =
                    plan->transform == MF_C2R ? MPSDataTypeFloat32 : MPSDataTypeComplexFloat32;

                id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputView.buffer;
                id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputView.buffer;
                MPSGraphTensorData *inputData = mf_tensor_data(inputBuffer, inputView.offset, inputShape, dataType);
                MPSGraphTensorData *outputData = mf_tensor_data(outputBuffer, outputView.offset, outputShape, outputDataType);
                if (inputData == nil || outputData == nil) {
                    mf_set_error(error_message, @"non-zero FFT buffer offsets require an Xcode 16+/macOS 15 build");
                    result = MF_ERROR_UNAVAILABLE;
                } else {
                    id<MTLCommandBuffer> commandBuffer =
                        (__bridge id<MTLCommandBuffer>)context.command_buffer;
                    MPSCommandBuffer *mpsCommandBuffer =
                        [MPSCommandBuffer commandBufferWithCommandBuffer:commandBuffer];
                    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
                        @{inputTensor: inputData};
                    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                        @{outputTensor: outputData};
                    [graph encodeToCommandBuffer:mpsCommandBuffer
                                           feeds:feeds
                                targetOperations:nil
                               resultsDictionary:results
                             executionDescriptor:nil];
                }
            }
        } @catch (NSException *exception) {
            mf_set_error(error_message, exception.reason);
            result = MF_ERROR_ENCODING;
        }

        char *endError = nullptr;
        const int endStatus = mr_end_external(&context, result != MF_SUCCESS, &endError);
        if (endStatus != MR_SUCCESS) {
            if (result == MF_SUCCESS) {
                mf_copy_runtime_error(error_message, endError);
                result = MF_ERROR_RUNTIME;
            } else if (endError != nullptr) {
                mr_free_error(endError);
            }
        }
        return result;
    }
}

extern "C" int mf_plan_destroy(void *opaquePlan, char **error_message) {
    (void)error_message;
    if (opaquePlan == nullptr) {
        return MF_SUCCESS;
    }
    @autoreleasepool {
        MFPlan *plan = (__bridge_transfer MFPlan *)opaquePlan;
        (void)plan;
    }
    return MF_SUCCESS;
}

extern "C" void mf_free_error(char *error_message) {
    free(error_message);
}
