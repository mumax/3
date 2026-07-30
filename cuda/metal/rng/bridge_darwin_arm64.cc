#include "bridge.h"
#include "../metal_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static const char mrgSourceText[] = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint philox_m0 = 0xD2511F53u;
constant uint philox_m1 = 0xCD9E8D57u;
constant uint philox_w0 = 0x9E3779B9u;
constant uint philox_w1 = 0xBB67AE85u;

struct RNGParams {
    uint seed_low;
    uint seed_high;
    uint counter_low;
    uint counter_high;
    uint count;
    float mean;
    float standard_deviation;
    uint padding;
};

inline uint multiply_high(uint lhs, uint rhs) {
    const uint lhs_low = lhs & 0xffffu;
    const uint lhs_high = lhs >> 16;
    const uint rhs_low = rhs & 0xffffu;
    const uint rhs_high = rhs >> 16;
    const uint low_product = lhs_low * rhs_low;
    const uint middle0 = lhs_high * rhs_low + (low_product >> 16);
    const uint middle1 = (middle0 & 0xffffu) + lhs_low * rhs_high;
    return lhs_high * rhs_high + (middle0 >> 16) + (middle1 >> 16);
}

inline uint4 philox_round(uint4 value, uint2 key) {
    const uint lo0 = philox_m0 * value.x;
    const uint hi0 = multiply_high(philox_m0, value.x);
    const uint lo1 = philox_m1 * value.z;
    const uint hi1 = multiply_high(philox_m1, value.z);
    return uint4(hi1 ^ value.y ^ key.x, lo1,
                 hi0 ^ value.w ^ key.y, lo0);
}

inline uint4 philox4x32_10(uint4 value, uint2 key) {
    for (uint round = 0; round < 10; ++round) {
        value = philox_round(value, key);
        key += uint2(philox_w0, philox_w1);
    }
    return value;
}

inline float uniform_open(uint value) {
    // Retain the upper 23 random bits, then select the midpoint of the
    // represented interval. 23 bits leave the half-integer exactly
    // representable, so the result is strictly inside (0,1) in float32.
    return (float(value >> 9) + 0.5f) * 0x1.0p-23f;
}

kernel void mumax3_philox_normal(
    device float *output [[buffer(0)]],
    constant RNGParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint base = index * 4u;
    if (base >= params.count) {
        return;
    }

    const uint counter_low = params.counter_low + index;
    const uint counter_high =
        params.counter_high + uint(counter_low < params.counter_low);
    const uint4 bits = philox4x32_10(
        uint4(counter_low, counter_high, 0u, 0u),
        uint2(params.seed_low, params.seed_high));

    const float u0 = uniform_open(bits.x);
    const float u1 = uniform_open(bits.y);
    const float u2 = uniform_open(bits.z);
    const float u3 = uniform_open(bits.w);
    const float radius0 = sqrt(-2.0f * log(u0));
    const float radius1 = sqrt(-2.0f * log(u2));
    const float angle0 = 6.2831853071795864769f * u1;
    const float angle1 = 6.2831853071795864769f * u3;
    const float4 normal = float4(radius0 * cos(angle0),
                                 radius0 * sin(angle0),
                                 radius1 * cos(angle1),
                                 radius1 * sin(angle1));

    for (uint lane = 0; lane < 4u && base + lane < params.count; ++lane) {
        output[base + lane] =
            params.mean + params.standard_deviation * normal[lane];
    }
}

struct RawParams {
    uint seed_low;
    uint seed_high;
    uint counter_low;
    uint counter_high;
    uint block_count;
    uint padding;
};

kernel void mumax3_philox_raw(
    device uint4 *output [[buffer(0)]],
    constant RawParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.block_count) {
        return;
    }
    const uint counter_low = params.counter_low + index;
    const uint counter_high =
        params.counter_high + uint(counter_low < params.counter_low);
    output[index] = philox4x32_10(
        uint4(counter_low, counter_high, 0u, 0u),
        uint2(params.seed_low, params.seed_high));
}
)MSL";

typedef struct mrg_parameters {
    uint32_t seed_low;
    uint32_t seed_high;
    uint32_t counter_low;
    uint32_t counter_high;
    uint32_t count;
    float mean;
    float standard_deviation;
    uint32_t padding;
} mrg_parameters;

typedef struct mrg_raw_parameters {
    uint32_t seed_low;
    uint32_t seed_high;
    uint32_t counter_low;
    uint32_t counter_high;
    uint32_t block_count;
    uint32_t padding;
} mrg_raw_parameters;

@interface MRGGenerator : NSObject {
@public
    uint64_t seed;
    uint64_t counter;
    id<MTLDevice> pipelineDevice;
    id<MTLComputePipelineState> normalPipeline;
    id<MTLComputePipelineState> rawPipeline;
}
@end

@implementation MRGGenerator
@end

static void mrg_set_error(char **destination, NSString *message) {
    if (destination == nullptr) {
        return;
    }
    const char *text = message == nil ? "unknown Metal RNG failure" : message.UTF8String;
    *destination = strdup(text == nullptr ? "unknown Metal RNG failure" : text);
}

static void mrg_copy_runtime_error(char **destination, char *runtimeError) {
    if (runtimeError == nullptr) {
        mrg_set_error(destination, @"unknown Metal runtime failure");
        return;
    }
    if (destination != nullptr) {
        *destination = strdup(runtimeError);
    }
    mr_free_error(runtimeError);
}

static BOOL mrg_ensure_pipeline(MRGGenerator *generator,
                                id<MTLDevice> device,
                                NSString **errorMessage) {
    if (generator->normalPipeline != nil && generator->rawPipeline != nil &&
        generator->pipelineDevice == device) {
        return YES;
    }

    NSString *source = [[NSString alloc] initWithBytes:mrgSourceText
                                                length:sizeof(mrgSourceText) - 1
                                              encoding:NSUTF8StringEncoding];
    NSError *libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&libraryError];
    if (library == nil) {
        if (errorMessage != nullptr) {
            *errorMessage = libraryError.localizedDescription;
        }
        return NO;
    }
    id<MTLFunction> normalFunction = [library newFunctionWithName:@"mumax3_philox_normal"];
    id<MTLFunction> rawFunction = [library newFunctionWithName:@"mumax3_philox_raw"];
    if (normalFunction == nil || rawFunction == nil) {
        if (errorMessage != nullptr) {
            *errorMessage = @"compiled RNG library is missing a Philox kernel";
        }
        return NO;
    }

    NSError *normalError = nil;
    id<MTLComputePipelineState> normalPipeline =
        [device newComputePipelineStateWithFunction:normalFunction error:&normalError];
    if (normalPipeline == nil) {
        if (errorMessage != nullptr) {
            *errorMessage = normalError.localizedDescription;
        }
        return NO;
    }
    NSError *rawError = nil;
    id<MTLComputePipelineState> rawPipeline =
        [device newComputePipelineStateWithFunction:rawFunction error:&rawError];
    if (rawPipeline == nil) {
        if (errorMessage != nullptr) {
            *errorMessage = rawError.localizedDescription;
        }
        return NO;
    }
    generator->pipelineDevice = device;
    generator->normalPipeline = normalPipeline;
    generator->rawPipeline = rawPipeline;
    return YES;
}

extern "C" void *mrg_generator_create(int32_t rng_type, char **error_message) {
    @autoreleasepool {
        // CURAND_RNG_PSEUDO_DEFAULT and CURAND_RNG_PSEUDO_XORWOW. The public
        // API names are retained, but both intentionally map to Philox.
        if (rng_type != 100 && rng_type != 101) {
            mrg_set_error(error_message, @"Metal supports pseudo-default/XORWOW API modes only; the sequence is Philox4x32-10");
            return nullptr;
        }
        MRGGenerator *generator = [MRGGenerator new];
        generator->seed = 0;
        generator->counter = 0;
        return (__bridge_retained void *)generator;
    }
}

extern "C" int mrg_generator_set_seed(void *opaqueGenerator,
                                       uint64_t newSeed,
                                       char **error_message) {
    if (opaqueGenerator == nullptr) {
        mrg_set_error(error_message, @"generator is null");
        return MRG_ERROR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        MRGGenerator *generator = (__bridge MRGGenerator *)opaqueGenerator;
        @synchronized(generator) {
            generator->seed = newSeed;
            generator->counter = 0;
        }
    }
    return MRG_SUCCESS;
}

extern "C" int mrg_generate_normal(void *opaqueGenerator,
                                    void *output,
                                    int64_t count,
                                    float mean,
                                    float standard_deviation,
                                    char **error_message) {
    @autoreleasepool {
        if (opaqueGenerator == nullptr || output == nullptr || count < 0 ||
            count > UINT32_MAX || standard_deviation < 0.0f) {
            mrg_set_error(error_message, @"invalid generator, output, count, or standard deviation");
            return MRG_ERROR_INVALID_ARGUMENT;
        }
        if (count == 0) {
            return MRG_SUCCESS;
        }
        if ((uint64_t)count > SIZE_MAX / sizeof(float)) {
            mrg_set_error(error_message, @"output byte count overflows size_t");
            return MRG_ERROR_INVALID_ARGUMENT;
        }

        MRGGenerator *generator = (__bridge MRGGenerator *)opaqueGenerator;
        @synchronized(generator) {
            mr_external_context context = {};
            char *runtimeError = nullptr;
            int runtimeStatus = mr_begin_external(&context, &runtimeError);
            if (runtimeStatus != MR_SUCCESS) {
                mrg_copy_runtime_error(error_message, runtimeError);
                return MRG_ERROR_RUNTIME;
            }

            int result = MRG_SUCCESS;
            id<MTLComputeCommandEncoder> encoder = nil;
            @try {
                id<MTLDevice> device = (__bridge id<MTLDevice>)context.device;
                NSString *pipelineError = nil;
                if (!mrg_ensure_pipeline(generator, device, &pipelineError)) {
                    mrg_set_error(error_message, pipelineError);
                    result = MRG_ERROR_COMPILE;
                }

                mr_buffer_view outputView = {};
                if (result == MRG_SUCCESS) {
                    runtimeStatus = mr_resolve_buffer_locked(
                        output, (size_t)count * sizeof(float), &outputView, &runtimeError);
                    if (runtimeStatus != MR_SUCCESS) {
                        mrg_copy_runtime_error(error_message, runtimeError);
                        result = MRG_ERROR_RUNTIME;
                    }
                }

                if (result == MRG_SUCCESS) {
                    id<MTLCommandBuffer> commandBuffer =
                        (__bridge id<MTLCommandBuffer>)context.command_buffer;
                    id<MTLBuffer> outputBuffer =
                        (__bridge id<MTLBuffer>)outputView.buffer;
                    encoder = [commandBuffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                    if (encoder == nil) {
                        mrg_set_error(error_message, @"failed to create RNG compute encoder");
                        result = MRG_ERROR_ENCODING;
                    } else {
                        mrg_parameters parameters = {
                            (uint32_t)generator->seed,
                            (uint32_t)(generator->seed >> 32),
                            (uint32_t)generator->counter,
                            (uint32_t)(generator->counter >> 32),
                            (uint32_t)count,
                            mean,
                            standard_deviation,
                            0,
                        };
                        [encoder setComputePipelineState:generator->normalPipeline];
                        [encoder setBuffer:outputBuffer offset:outputView.offset atIndex:0];
                        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];

                        const NSUInteger threadCount = ((NSUInteger)count + 3u) / 4u;
                        const NSUInteger executionWidth =
                            MAX((NSUInteger)1, generator->normalPipeline.threadExecutionWidth);
                        const NSUInteger threadsPerGroup =
                            MIN((NSUInteger)256,
                                MIN(generator->normalPipeline.maxTotalThreadsPerThreadgroup,
                                    executionWidth * 8u));
                        [encoder dispatchThreads:MTLSizeMake(threadCount, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                        [encoder endEncoding];
                        encoder = nil;
                        generator->counter += (uint64_t)threadCount;
                    }
                }
            } @catch (NSException *exception) {
                if (encoder != nil) {
                    [encoder endEncoding];
                    encoder = nil;
                }
                mrg_set_error(error_message, exception.reason);
                result = MRG_ERROR_ENCODING;
            }

            char *endError = nullptr;
            const int endStatus = mr_end_external(&context, result != MRG_SUCCESS, &endError);
            if (endStatus != MR_SUCCESS) {
                if (result == MRG_SUCCESS) {
                    mrg_copy_runtime_error(error_message, endError);
                    result = MRG_ERROR_RUNTIME;
                } else if (endError != nullptr) {
                    mr_free_error(endError);
                }
            }
            return result;
        }
    }
}

extern "C" int mrg_generate_raw(void *opaqueGenerator,
                                 void *output,
                                 int64_t block_count,
                                 char **error_message) {
    @autoreleasepool {
        if (opaqueGenerator == nullptr || output == nullptr ||
            block_count < 0 || block_count > UINT32_MAX) {
            mrg_set_error(error_message, @"invalid generator, output, or Philox block count");
            return MRG_ERROR_INVALID_ARGUMENT;
        }
        if (block_count == 0) {
            return MRG_SUCCESS;
        }
        if ((uint64_t)block_count > SIZE_MAX / (4u * sizeof(uint32_t))) {
            mrg_set_error(error_message, @"raw Philox byte count overflows size_t");
            return MRG_ERROR_INVALID_ARGUMENT;
        }

        MRGGenerator *generator = (__bridge MRGGenerator *)opaqueGenerator;
        @synchronized(generator) {
            mr_external_context context = {};
            char *runtimeError = nullptr;
            int runtimeStatus = mr_begin_external(&context, &runtimeError);
            if (runtimeStatus != MR_SUCCESS) {
                mrg_copy_runtime_error(error_message, runtimeError);
                return MRG_ERROR_RUNTIME;
            }

            int result = MRG_SUCCESS;
            id<MTLComputeCommandEncoder> encoder = nil;
            @try {
                id<MTLDevice> device = (__bridge id<MTLDevice>)context.device;
                NSString *pipelineError = nil;
                if (!mrg_ensure_pipeline(generator, device, &pipelineError)) {
                    mrg_set_error(error_message, pipelineError);
                    result = MRG_ERROR_COMPILE;
                }

                mr_buffer_view outputView = {};
                if (result == MRG_SUCCESS) {
                    const size_t bytes =
                        (size_t)block_count * 4u * sizeof(uint32_t);
                    runtimeStatus = mr_resolve_buffer_locked(
                        output, bytes, &outputView, &runtimeError);
                    if (runtimeStatus != MR_SUCCESS) {
                        mrg_copy_runtime_error(error_message, runtimeError);
                        result = MRG_ERROR_RUNTIME;
                    }
                }

                if (result == MRG_SUCCESS) {
                    id<MTLCommandBuffer> commandBuffer =
                        (__bridge id<MTLCommandBuffer>)context.command_buffer;
                    id<MTLBuffer> outputBuffer =
                        (__bridge id<MTLBuffer>)outputView.buffer;
                    encoder = [commandBuffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                    if (encoder == nil) {
                        mrg_set_error(error_message, @"failed to create raw Philox compute encoder");
                        result = MRG_ERROR_ENCODING;
                    } else {
                        mrg_raw_parameters parameters = {
                            (uint32_t)generator->seed,
                            (uint32_t)(generator->seed >> 32),
                            (uint32_t)generator->counter,
                            (uint32_t)(generator->counter >> 32),
                            (uint32_t)block_count,
                            0,
                        };
                        [encoder setComputePipelineState:generator->rawPipeline];
                        [encoder setBuffer:outputBuffer offset:outputView.offset atIndex:0];
                        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
                        const NSUInteger executionWidth =
                            MAX((NSUInteger)1, generator->rawPipeline.threadExecutionWidth);
                        const NSUInteger threadsPerGroup =
                            MIN((NSUInteger)256,
                                MIN(generator->rawPipeline.maxTotalThreadsPerThreadgroup,
                                    executionWidth * 8u));
                        [encoder dispatchThreads:MTLSizeMake((NSUInteger)block_count, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                        [encoder endEncoding];
                        encoder = nil;
                        generator->counter += (uint64_t)block_count;
                    }
                }
            } @catch (NSException *exception) {
                if (encoder != nil) {
                    [encoder endEncoding];
                    encoder = nil;
                }
                mrg_set_error(error_message, exception.reason);
                result = MRG_ERROR_ENCODING;
            }

            char *endError = nullptr;
            const int endStatus = mr_end_external(&context, result != MRG_SUCCESS, &endError);
            if (endStatus != MR_SUCCESS) {
                if (result == MRG_SUCCESS) {
                    mrg_copy_runtime_error(error_message, endError);
                    result = MRG_ERROR_RUNTIME;
                } else if (endError != nullptr) {
                    mr_free_error(endError);
                }
            }
            return result;
        }
    }
}

extern "C" void mrg_free_error(char *error_message) {
    free(error_message);
}
