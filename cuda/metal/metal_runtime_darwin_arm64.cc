#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_runtime.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace {

struct Allocation {
    uintptr_t base;
    size_t length;
    void *retained_buffer;
};

std::mutex runtime_mutex;
id<MTLDevice> device = nil;
id<MTLCommandQueue> queue = nil;
id<MTLCommandBuffer> current_command_buffer = nil;
id<MTLBuffer> zero_buffer = nil;
NSMutableArray<id<MTLLibrary>> *libraries = nil;
NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines = nil;
NSMutableArray<id<MTLCommandBuffer>> *submitted_command_buffers = nil;
std::map<uintptr_t, Allocation> allocations;
uint64_t allocated_bytes = 0;
uint64_t peak_allocated_bytes = 0;
uint64_t command_buffer_sequence = 0;
uint32_t encoded_operation_count = 0;
uint64_t external_sequence = 0;
uint64_t active_external_token = 0;

constexpr size_t zero_buffer_bytes = 4096;
constexpr size_t max_buffer_arguments = 31;
constexpr uint32_t max_operations_per_command_buffer = 256;
constexpr NSUInteger max_pending_command_buffers = 8;
constexpr char runtime_kernel_source[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void mumax3_runtime_fill_u32(
    device uint *destination [[buffer(0)]],
    constant uint &value [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        destination[index] = value;
    }
}
)METAL";

int fail(int code, char **error_message, const std::string &message) {
    if (error_message != nullptr) {
        *error_message = ::strdup(message.c_str());
    }
    return code;
}

std::string describe(NSError *error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    NSString *description = error.localizedDescription;
    if (description == nil) {
        description = error.description;
    }
    return description == nil ? "unknown Metal error" : std::string(description.UTF8String);
}

void releaseAllocation(const Allocation &allocation) {
    if (allocation.retained_buffer != nullptr) {
        id released =
            CFBridgingRelease(allocation.retained_buffer);
        (void)released;
    }
}

int initializeUnlocked(std::string &error) {
    if (device != nil) {
        return MR_SUCCESS;
    }

    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        error = "no Metal device is available";
        return MR_ERROR_UNAVAILABLE;
    }
    queue = [device newCommandQueue];
    if (queue == nil) {
        device = nil;
        error = "failed to create the Metal command queue";
        return MR_ERROR_UNAVAILABLE;
    }

    zero_buffer = [device newBufferWithLength:zero_buffer_bytes
                                      options:MTLResourceStorageModeShared];
    if (zero_buffer == nil || zero_buffer.contents == nullptr) {
        zero_buffer = nil;
        queue = nil;
        device = nil;
        error = "failed to allocate the Metal optional-argument zero buffer";
        return MR_ERROR_OUT_OF_MEMORY;
    }
    std::memset(zero_buffer.contents, 0, zero_buffer_bytes);
    zero_buffer.label = @"MuMax3 optional zero buffer";

    libraries = [[NSMutableArray alloc] init];
    pipelines = [[NSMutableDictionary alloc] init];
    submitted_command_buffers = [[NSMutableArray alloc] init];
    NSError *runtime_library_error = nil;
    NSString *runtime_source =
        [[NSString alloc] initWithBytes:runtime_kernel_source
                                length:sizeof(runtime_kernel_source) - 1
                              encoding:NSUTF8StringEncoding];
    MTLCompileOptions *runtime_options = [[MTLCompileOptions alloc] init];
    runtime_options.fastMathEnabled = NO;
    id<MTLLibrary> runtime_library =
        [device newLibraryWithSource:runtime_source
                             options:runtime_options
                               error:&runtime_library_error];
    if (runtime_library == nil) {
        pipelines = nil;
        libraries = nil;
        submitted_command_buffers = nil;
        zero_buffer = nil;
        queue = nil;
        device = nil;
        error = "failed to compile the Metal runtime kernels: " +
                describe(runtime_library_error);
        return MR_ERROR_COMPILE;
    }
    [libraries addObject:runtime_library];
    id<MTLLibrary> default_library = [device newDefaultLibrary];
    if (default_library != nil) {
        [libraries addObject:default_library];
    }
    return MR_SUCCESS;
}

id<MTLCommandBuffer> commandBufferUnlocked(std::string &error) {
    if (current_command_buffer != nil) {
        return current_command_buffer;
    }
    current_command_buffer = [queue commandBuffer];
    if (current_command_buffer == nil) {
        error = "failed to create a Metal command buffer";
        return nil;
    }
    ++command_buffer_sequence;
    current_command_buffer.label =
        [NSString stringWithFormat:@"MuMax3 batch %llu", command_buffer_sequence];
    return current_command_buffer;
}

int checkSubmittedCommandUnlocked(id<MTLCommandBuffer> command_buffer,
                                  std::string &error) {
    if (command_buffer == nil) {
        return MR_SUCCESS;
    }
    if (command_buffer.status == MTLCommandBufferStatusError) {
        error = "Metal command buffer failed: " + describe(command_buffer.error);
        return MR_ERROR_COMMAND;
    }
    return MR_SUCCESS;
}

int submitUnlocked(bool wait, std::string &error) {
    if (current_command_buffer != nil) {
        id<MTLCommandBuffer> submitting = current_command_buffer;
        current_command_buffer = nil;
        encoded_operation_count = 0;
        [submitting commit];
        [submitted_command_buffers addObject:submitting];
    }

    if (submitted_command_buffers.count == 0) {
        return MR_SUCCESS;
    }

    if (!wait) {
        if (submitted_command_buffers.count <=
            max_pending_command_buffers) {
            return MR_SUCCESS;
        }
        /*
         * Bound retained command-buffer metadata and provide queue
         * backpressure without changing stream ordering. Waiting for the
         * oldest item still leaves up to max_pending_command_buffers batches
         * executing asynchronously.
         */
        id<MTLCommandBuffer> oldest = submitted_command_buffers.firstObject;
        [oldest waitUntilCompleted];
        int status = checkSubmittedCommandUnlocked(oldest, error);
        [submitted_command_buffers removeObjectAtIndex:0];
        return status;
    }

    int first_status = MR_SUCCESS;
    std::string first_error;
    for (id<MTLCommandBuffer> submitted in submitted_command_buffers) {
        [submitted waitUntilCompleted];
        std::string submitted_error;
        int status =
            checkSubmittedCommandUnlocked(submitted, submitted_error);
        if (status != MR_SUCCESS && first_status == MR_SUCCESS) {
            first_status = status;
            first_error = submitted_error;
        }
    }
    [submitted_command_buffers removeAllObjects];
    if (first_status != MR_SUCCESS) {
        error = first_error;
    }
    return first_status;
}

int operationEncodedUnlocked(std::string &error) {
    ++encoded_operation_count;
    if (encoded_operation_count < max_operations_per_command_buffer) {
        return MR_SUCCESS;
    }
    /*
     * Command buffers are ordered by the single command queue. Committing this
     * batch without waiting bounds encoder metadata while preserving CUDA
     * stream-0 ordering for the next batch.
     */
    return submitUnlocked(false, error);
}

int resolveBufferUnlocked(const void *pointer,
                          size_t minimum_bytes,
                          __strong id<MTLBuffer> &buffer,
                          size_t &offset,
                          size_t &available,
                          std::string &error) {
    if (pointer == nullptr) {
        buffer = zero_buffer;
        offset = 0;
        available = zero_buffer_bytes;
        return MR_SUCCESS;
    }

    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    auto candidate = allocations.upper_bound(address);
    if (candidate == allocations.begin()) {
        error = "pointer is not owned by the MuMax3 Metal allocator";
        return MR_ERROR_INVALID_ARGUMENT;
    }
    --candidate;
    const Allocation &allocation = candidate->second;
    if (address < allocation.base) {
        error = "pointer precedes its Metal allocation";
        return MR_ERROR_INVALID_ARGUMENT;
    }
    uintptr_t relative = address - allocation.base;
    if (relative >= allocation.length) {
        error = "pointer is outside every live Metal allocation";
        return MR_ERROR_INVALID_ARGUMENT;
    }

    available = allocation.length - static_cast<size_t>(relative);
    if (minimum_bytes > available) {
        std::ostringstream stream;
        stream << "buffer span of " << minimum_bytes
               << " bytes exceeds the allocation remainder of " << available
               << " bytes";
        error = stream.str();
        return MR_ERROR_INVALID_ARGUMENT;
    }

    buffer = (__bridge id<MTLBuffer>)allocation.retained_buffer;
    offset = static_cast<size_t>(relative);
    return MR_SUCCESS;
}

id<MTLComputePipelineState> pipelineUnlocked(const char *name,
                                             int &status,
                                             std::string &error) {
    NSString *kernel_name = [NSString stringWithUTF8String:name];
    if (kernel_name == nil) {
        status = MR_ERROR_INVALID_ARGUMENT;
        error = "kernel name is not valid UTF-8";
        return nil;
    }

    id<MTLComputePipelineState> cached = pipelines[kernel_name];
    if (cached != nil) {
        status = MR_SUCCESS;
        return cached;
    }

    id<MTLFunction> function = nil;
    for (NSInteger index = libraries.count - 1; index >= 0; --index) {
        function = [libraries[index] newFunctionWithName:kernel_name];
        if (function != nil) {
            break;
        }
    }
    if (function == nil) {
        status = MR_ERROR_NOT_FOUND;
        std::ostringstream stream;
        stream << "Metal kernel '" << name << "' was not found in "
               << libraries.count << " registered libraries";
        error = stream.str();
        return nil;
    }

    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (pipeline == nil) {
        status = MR_ERROR_PIPELINE;
        error = "failed to create pipeline for '" + std::string(name) +
                "': " + describe(pipeline_error);
        return nil;
    }
    pipelines[kernel_name] = pipeline;
    status = MR_SUCCESS;
    return pipeline;
}

bool rangesOverlap(size_t first_offset,
                   size_t second_offset,
                   size_t length) {
    if (length == 0) {
        return false;
    }
    return first_offset < second_offset + length &&
           second_offset < first_offset + length;
}

int checkInitialized(char **error_message) {
    std::string error;
    int status = initializeUnlocked(error);
    if (status != MR_SUCCESS) {
        return fail(status, error_message, error);
    }
    return MR_SUCCESS;
}

}  // namespace

int mr_initialize(char **error_message) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(runtime_mutex);
        return checkInitialized(error_message);
    }
}

int mr_shutdown(char **error_message) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(runtime_mutex);
        if (device == nil) {
            return MR_SUCCESS;
        }

        std::string error;
        int status = submitUnlocked(true, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, error);
        }

        for (const auto &entry : allocations) {
            releaseAllocation(entry.second);
        }
        allocations.clear();
        allocated_bytes = 0;
        peak_allocated_bytes = 0;
        [pipelines removeAllObjects];
        [libraries removeAllObjects];
        pipelines = nil;
        libraries = nil;
        zero_buffer = nil;
        [submitted_command_buffers removeAllObjects];
        submitted_command_buffers = nil;
        current_command_buffer = nil;
        encoded_operation_count = 0;
        queue = nil;
        device = nil;
        active_external_token = 0;
        return MR_SUCCESS;
    }
}

int mr_get_device_info(mr_device_info *info, char **error_message) {
    @autoreleasepool {
        if (info == nullptr) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "device info output is null");
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        std::memset(info, 0, sizeof(*info));
        const char *name = device.name.UTF8String;
        if (name != nullptr) {
            std::strncpy(info->name, name, sizeof(info->name) - 1);
        }
        info->unified_memory = device.hasUnifiedMemory ? 1 : 0;
        MTLSize maximum = device.maxThreadsPerThreadgroup;
        info->max_threads_x = maximum.width;
        info->max_threads_y = maximum.height;
        info->max_threads_z = maximum.depth;
        info->recommended_max_working_set_size =
            device.recommendedMaxWorkingSetSize;
        info->current_allocated_size = device.currentAllocatedSize;
        info->tracked_allocation_size = allocated_bytes;
        info->tracked_peak_allocation_size = peak_allocated_bytes;
        return MR_SUCCESS;
    }
}

int mr_register_source(const void *source,
                       size_t length,
                       char **error_message) {
    @autoreleasepool {
        if (source == nullptr || length == 0) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "Metal source is empty");
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        NSData *data = [NSData dataWithBytes:source length:length];
        NSString *text = [[NSString alloc] initWithData:data
                                               encoding:NSUTF8StringEncoding];
        if (text == nil) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "Metal source is not valid UTF-8");
        }
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO;
        NSError *compile_error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:text
                                 options:options
                                   error:&compile_error];
        if (library == nil) {
            return fail(MR_ERROR_COMPILE,
                        error_message,
                        "Metal source compilation failed: " +
                            describe(compile_error));
        }
        [libraries addObject:library];
        [pipelines removeAllObjects];
        return MR_SUCCESS;
    }
}

int mr_register_library(const void *library,
                        size_t length,
                        char **error_message) {
    @autoreleasepool {
        if (library == nullptr || length == 0) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "metallib payload is empty");
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        void *owned_copy = std::malloc(length);
        if (owned_copy == nullptr) {
            return fail(MR_ERROR_OUT_OF_MEMORY,
                        error_message,
                        "failed to copy the metallib payload");
        }
        std::memcpy(owned_copy, library, length);
        dispatch_data_t data =
            dispatch_data_create(owned_copy,
                                 length,
                                 dispatch_get_global_queue(
                                     QOS_CLASS_USER_INITIATED, 0),
                                 DISPATCH_DATA_DESTRUCTOR_FREE);
        NSError *library_error = nil;
        id<MTLLibrary> metal_library =
            [device newLibraryWithData:data error:&library_error];
        if (metal_library == nil) {
            return fail(MR_ERROR_COMPILE,
                        error_message,
                        "metallib loading failed: " +
                            describe(library_error));
        }
        [libraries addObject:metal_library];
        [pipelines removeAllObjects];
        return MR_SUCCESS;
    }
}

int mr_alloc(size_t bytes, void **pointer, char **error_message) {
    @autoreleasepool {
        if (pointer == nullptr || bytes == 0) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "allocation output and size must be non-zero");
        }
        *pointer = nullptr;
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        id<MTLBuffer> buffer =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buffer == nil || buffer.contents == nullptr) {
            return fail(MR_ERROR_OUT_OF_MEMORY,
                        error_message,
                        "Metal shared-buffer allocation failed");
        }
        uintptr_t base = reinterpret_cast<uintptr_t>(buffer.contents);
        if (base > std::numeric_limits<uintptr_t>::max() - bytes) {
            return fail(MR_ERROR_INTERNAL,
                        error_message,
                        "Metal allocation address range overflowed uintptr_t");
        }
        if (allocations.find(base) != allocations.end()) {
            return fail(MR_ERROR_INTERNAL,
                        error_message,
                        "Metal returned an already-registered allocation address");
        }

        Allocation allocation{
            base,
            bytes,
            (__bridge_retained void *)buffer,
        };
        allocations.emplace(base, allocation);
        allocated_bytes += bytes;
        peak_allocated_bytes = std::max(peak_allocated_bytes, allocated_bytes);
        *pointer = reinterpret_cast<void *>(base);
        return MR_SUCCESS;
    }
}

int mr_free(void *pointer, char **error_message) {
    @autoreleasepool {
        if (pointer == nullptr) {
            return MR_SUCCESS;
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        uintptr_t base = reinterpret_cast<uintptr_t>(pointer);
        auto allocation = allocations.find(base);
        if (allocation == allocations.end()) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "free requires the base address of a live Metal allocation");
        }

        std::string error;
        status = submitUnlocked(true, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, error);
        }
        allocated_bytes -= allocation->second.length;
        releaseAllocation(allocation->second);
        allocations.erase(allocation);
        return MR_SUCCESS;
    }
}

int mr_get_address_range(const void *pointer,
                         void **base,
                         size_t *bytes,
                         char **error_message) {
    @autoreleasepool {
        if (pointer == nullptr || base == nullptr || bytes == nullptr) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "allocation range query contains a null argument");
        }
        *base = nullptr;
        *bytes = 0;
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        std::string error;
        id<MTLBuffer> buffer = nil;
        size_t offset = 0;
        size_t available = 0;
        status = resolveBufferUnlocked(
            pointer, 0, buffer, offset, available, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, error);
        }
        uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
        *base = reinterpret_cast<void *>(address - offset);
        *bytes = offset + available;
        return MR_SUCCESS;
    }
}

int mr_copy(void *dst,
            const void *src,
            size_t bytes,
            char **error_message) {
    @autoreleasepool {
        if (bytes == 0) {
            return MR_SUCCESS;
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        std::string error;
        id<MTLBuffer> dst_buffer = nil;
        id<MTLBuffer> src_buffer = nil;
        size_t dst_offset = 0;
        size_t src_offset = 0;
        size_t dst_available = 0;
        size_t src_available = 0;
        status = resolveBufferUnlocked(
            dst, bytes, dst_buffer, dst_offset, dst_available, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, "copy destination: " + error);
        }
        status = resolveBufferUnlocked(
            src, bytes, src_buffer, src_offset, src_available, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, "copy source: " + error);
        }

        if (dst_buffer == src_buffer &&
            rangesOverlap(dst_offset, src_offset, bytes)) {
            status = submitUnlocked(true, error);
            if (status != MR_SUCCESS) {
                return fail(status, error_message, error);
            }
            std::memmove(static_cast<uint8_t *>(dst_buffer.contents) + dst_offset,
                         static_cast<const uint8_t *>(src_buffer.contents) +
                             src_offset,
                         bytes);
            return MR_SUCCESS;
        }

        id<MTLCommandBuffer> command_buffer = commandBufferUnlocked(error);
        if (command_buffer == nil) {
            return fail(MR_ERROR_COMMAND, error_message, error);
        }
        id<MTLBlitCommandEncoder> encoder =
            [command_buffer blitCommandEncoder];
        if (encoder == nil) {
            return fail(MR_ERROR_COMMAND,
                        error_message,
                        "failed to create a Metal blit encoder");
        }
        [encoder copyFromBuffer:src_buffer
                  sourceOffset:src_offset
                      toBuffer:dst_buffer
             destinationOffset:dst_offset
                          size:bytes];
        [encoder endEncoding];
        status = operationEncodedUnlocked(error);
        return status == MR_SUCCESS
                   ? MR_SUCCESS
                   : fail(status, error_message, error);
    }
}

int mr_copy_to_device(void *dst,
                      const void *src,
                      size_t bytes,
                      char **error_message) {
    @autoreleasepool {
        if (bytes == 0) {
            return MR_SUCCESS;
        }
        if (src == nullptr) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "host-to-device source is null");
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        std::string error;
        id<MTLBuffer> dst_buffer = nil;
        size_t dst_offset = 0;
        size_t dst_available = 0;
        status = resolveBufferUnlocked(
            dst, bytes, dst_buffer, dst_offset, dst_available, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, "copy destination: " + error);
        }
        status = submitUnlocked(true, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, error);
        }
        std::memmove(static_cast<uint8_t *>(dst_buffer.contents) + dst_offset,
                     src,
                     bytes);
        return MR_SUCCESS;
    }
}

int mr_copy_to_host(void *dst,
                    const void *src,
                    size_t bytes,
                    char **error_message) {
    @autoreleasepool {
        if (bytes == 0) {
            return MR_SUCCESS;
        }
        if (dst == nullptr) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "device-to-host destination is null");
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        std::string error;
        id<MTLBuffer> src_buffer = nil;
        size_t src_offset = 0;
        size_t src_available = 0;
        status = resolveBufferUnlocked(
            src, bytes, src_buffer, src_offset, src_available, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, "copy source: " + error);
        }
        status = submitUnlocked(true, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, error);
        }
        std::memmove(dst,
                     static_cast<const uint8_t *>(src_buffer.contents) +
                         src_offset,
                     bytes);
        return MR_SUCCESS;
    }
}

int mr_fill(void *dst,
            uint8_t value,
            size_t bytes,
            char **error_message) {
    @autoreleasepool {
        if (bytes == 0) {
            return MR_SUCCESS;
        }
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        std::string error;
        id<MTLBuffer> dst_buffer = nil;
        size_t dst_offset = 0;
        size_t dst_available = 0;
        status = resolveBufferUnlocked(
            dst, bytes, dst_buffer, dst_offset, dst_available, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, "fill destination: " + error);
        }
        id<MTLCommandBuffer> command_buffer = commandBufferUnlocked(error);
        if (command_buffer == nil) {
            return fail(MR_ERROR_COMMAND, error_message, error);
        }
        id<MTLBlitCommandEncoder> encoder =
            [command_buffer blitCommandEncoder];
        if (encoder == nil) {
            return fail(MR_ERROR_COMMAND,
                        error_message,
                        "failed to create a Metal blit encoder");
        }
        [encoder fillBuffer:dst_buffer
                      range:NSMakeRange(dst_offset, bytes)
                      value:value];
        [encoder endEncoding];
        status = operationEncodedUnlocked(error);
        return status == MR_SUCCESS
                   ? MR_SUCCESS
                   : fail(status, error_message, error);
    }
}

int mr_fill_u32(void *dst,
                uint32_t value,
                size_t count,
                char **error_message) {
    @autoreleasepool {
        if (count == 0) {
            return MR_SUCCESS;
        }
        if (count > std::numeric_limits<uint32_t>::max() ||
            count > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "uint32 fill count exceeds the Metal kernel limit");
        }

        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        const size_t byte_count = count * sizeof(uint32_t);
        std::string error;
        id<MTLBuffer> dst_buffer = nil;
        size_t dst_offset = 0;
        size_t dst_available = 0;
        status = resolveBufferUnlocked(
            dst, byte_count, dst_buffer, dst_offset, dst_available, error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, "fill destination: " + error);
        }

        int pipeline_status = MR_SUCCESS;
        id<MTLComputePipelineState> pipeline =
            pipelineUnlocked("mumax3_runtime_fill_u32",
                             pipeline_status,
                             error);
        if (pipeline == nil) {
            return fail(pipeline_status, error_message, error);
        }
        id<MTLCommandBuffer> command_buffer = commandBufferUnlocked(error);
        if (command_buffer == nil) {
            return fail(MR_ERROR_COMMAND, error_message, error);
        }
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return fail(MR_ERROR_COMMAND,
                        error_message,
                        "failed to create a Metal uint32 fill encoder");
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:dst_buffer offset:dst_offset atIndex:0];
        [encoder setBytes:&value length:sizeof(value) atIndex:1];
        uint32_t kernel_count = static_cast<uint32_t>(count);
        [encoder setBytes:&kernel_count length:sizeof(kernel_count) atIndex:2];

        NSUInteger width =
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        status = operationEncodedUnlocked(error);
        return status == MR_SUCCESS
                   ? MR_SUCCESS
                   : fail(status, error_message, error);
    }
}

int mr_launch(const char *name,
              mr_grid grid,
              const mr_arg *args,
              size_t arg_count,
              char **error_message) {
    @autoreleasepool {
        if (name == nullptr || name[0] == '\0') {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "kernel name is empty");
        }
        if (arg_count > 0 && args == nullptr) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "kernel argument array is null");
        }
        if (arg_count > max_buffer_arguments) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "kernel uses more than 31 Metal buffer-table arguments");
        }
        if (grid.block_x == 0 || grid.block_y == 0 || grid.block_z == 0) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "threadgroup dimensions must be non-zero");
        }
        if (grid.grid_x == 0 || grid.grid_y == 0 || grid.grid_z == 0) {
            return MR_SUCCESS;
        }

        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }

        std::string error;
        id<MTLComputePipelineState> pipeline =
            pipelineUnlocked(name, status, error);
        if (pipeline == nil) {
            return fail(status, error_message, error);
        }

        uint64_t block_xy =
            static_cast<uint64_t>(grid.block_x) * grid.block_y;
        uint64_t block_threads = block_xy * grid.block_z;
        MTLSize device_limit = device.maxThreadsPerThreadgroup;
        if (grid.block_x > device_limit.width ||
            grid.block_y > device_limit.height ||
            grid.block_z > device_limit.depth ||
            block_threads > pipeline.maxTotalThreadsPerThreadgroup) {
            std::ostringstream stream;
            stream << "threadgroup (" << grid.block_x << ", "
                   << grid.block_y << ", " << grid.block_z
                   << ") exceeds the device or pipeline limit of "
                   << pipeline.maxTotalThreadsPerThreadgroup
                   << " total threads";
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        stream.str());
        }

        MTLSize threads_per_threadgroup =
            MTLSizeMake(grid.block_x, grid.block_y, grid.block_z);
        MTLSize threads_per_grid =
            MTLSizeMake(static_cast<NSUInteger>(grid.grid_x) * grid.block_x,
                        static_cast<NSUInteger>(grid.grid_y) * grid.block_y,
                        static_cast<NSUInteger>(grid.grid_z) * grid.block_z);

        id<MTLCommandBuffer> command_buffer = commandBufferUnlocked(error);
        if (command_buffer == nil) {
            return fail(MR_ERROR_COMMAND, error_message, error);
        }
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return fail(MR_ERROR_COMMAND,
                        error_message,
                        "failed to create a Metal compute encoder");
        }
        encoder.label = [NSString stringWithUTF8String:name];
        [encoder setComputePipelineState:pipeline];

        for (size_t index = 0; index < arg_count; ++index) {
            const mr_arg &argument = args[index];
            switch (argument.kind) {
                case MR_ARG_BUFFER: {
                    id<MTLBuffer> buffer = nil;
                    size_t offset = 0;
                    size_t available = 0;
                    status = resolveBufferUnlocked(argument.buffer,
                                                   argument.size,
                                                   buffer,
                                                   offset,
                                                   available,
                                                   error);
                    if (status != MR_SUCCESS) {
                        [encoder endEncoding];
                        std::ostringstream stream;
                        stream << "kernel '" << name << "' argument "
                               << index << ": " << error;
                        return fail(status, error_message, stream.str());
                    }
                    [encoder setBuffer:buffer
                                offset:offset
                               atIndex:index];
                    break;
                }
                case MR_ARG_FLOAT32:
                case MR_ARG_INT32:
                case MR_ARG_UINT32:
                    if (argument.size != 4) {
                        [encoder endEncoding];
                        return fail(MR_ERROR_INVALID_ARGUMENT,
                                    error_message,
                                    "32-bit scalar argument has an invalid size");
                    }
                    [encoder setBytes:&argument.bits
                               length:4
                              atIndex:index];
                    break;
                case MR_ARG_UINT8:
                    if (argument.size != 1) {
                        [encoder endEncoding];
                        return fail(MR_ERROR_INVALID_ARGUMENT,
                                    error_message,
                                    "8-bit scalar argument has an invalid size");
                    }
                    [encoder setBytes:&argument.bits
                               length:1
                              atIndex:index];
                    break;
                case MR_ARG_FLOAT64:
                case MR_ARG_INT64:
                case MR_ARG_UINT64:
                    if (argument.size != 8) {
                        [encoder endEncoding];
                        return fail(MR_ERROR_INVALID_ARGUMENT,
                                    error_message,
                                    "64-bit scalar argument has an invalid size");
                    }
                    [encoder setBytes:&argument.bits
                               length:8
                              atIndex:index];
                    break;
                default:
                    [encoder endEncoding];
                    return fail(MR_ERROR_INVALID_ARGUMENT,
                                error_message,
                                "kernel argument has an unknown kind");
            }
        }

        [encoder dispatchThreads:threads_per_grid
           threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
        status = operationEncodedUnlocked(error);
        return status == MR_SUCCESS
                   ? MR_SUCCESS
                   : fail(status, error_message, error);
    }
}

int mr_flush(char **error_message) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }
        std::string error;
        status = submitUnlocked(false, error);
        return status == MR_SUCCESS
                   ? MR_SUCCESS
                   : fail(status, error_message, error);
    }
}

int mr_synchronize(char **error_message) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(runtime_mutex);
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            return status;
        }
        std::string error;
        status = submitUnlocked(true, error);
        return status == MR_SUCCESS
                   ? MR_SUCCESS
                   : fail(status, error_message, error);
    }
}

int mr_begin_external(mr_external_context *context,
                      char **error_message) {
    if (context == nullptr) {
        return fail(MR_ERROR_INVALID_ARGUMENT,
                    error_message,
                    "external Metal context output is null");
    }
    std::memset(context, 0, sizeof(*context));
    runtime_mutex.lock();
    @autoreleasepool {
        int status = checkInitialized(error_message);
        if (status != MR_SUCCESS) {
            runtime_mutex.unlock();
            return status;
        }
        std::string error;
        id<MTLCommandBuffer> command_buffer = commandBufferUnlocked(error);
        if (command_buffer == nil) {
            runtime_mutex.unlock();
            return fail(MR_ERROR_COMMAND, error_message, error);
        }
        active_external_token = ++external_sequence;
        context->device = (__bridge void *)device;
        context->queue = (__bridge void *)queue;
        context->command_buffer = (__bridge void *)command_buffer;
        context->token = active_external_token;
        return MR_SUCCESS;
    }
}

int mr_resolve_buffer_locked(const void *pointer,
                             size_t minimum_bytes,
                             mr_buffer_view *view,
                             char **error_message) {
    @autoreleasepool {
        if (view == nullptr) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "external Metal buffer view output is null");
        }
        std::memset(view, 0, sizeof(*view));
        if (active_external_token == 0) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "buffer resolution requires an active external context");
        }

        std::string error;
        id<MTLBuffer> buffer = nil;
        size_t offset = 0;
        size_t available = 0;
        int status = resolveBufferUnlocked(pointer,
                                           minimum_bytes,
                                           buffer,
                                           offset,
                                           available,
                                           error);
        if (status != MR_SUCCESS) {
            return fail(status, error_message, error);
        }
        view->buffer = (__bridge void *)buffer;
        view->offset = offset;
        view->length = available;
        return MR_SUCCESS;
    }
}

int mr_end_external(mr_external_context *context,
                    int encoder_failed,
                    char **error_message) {
    @autoreleasepool {
        if (context == nullptr || context->token == 0 ||
            context->token != active_external_token) {
            return fail(MR_ERROR_INVALID_ARGUMENT,
                        error_message,
                        "external Metal context token is invalid");
        }
        std::memset(context, 0, sizeof(*context));
        active_external_token = 0;
        std::string error;
        int status = MR_SUCCESS;
        if (encoder_failed == 0) {
            status = operationEncodedUnlocked(error);
        }
        runtime_mutex.unlock();
        if (encoder_failed != 0) {
            return fail(MR_ERROR_COMMAND,
                        error_message,
                        "external Metal encoder reported a failure; "
                        "the current batch is retained for synchronization");
        }
        return status == MR_SUCCESS
                   ? MR_SUCCESS
                   : fail(status, error_message, error);
    }
}

void mr_free_error(char *error_message) {
    std::free(error_message);
}
