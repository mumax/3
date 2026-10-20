#ifndef MUMAX3_METAL_RUNTIME_H
#define MUMAX3_METAL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mr_status {
    MR_SUCCESS = 0,
    MR_ERROR_UNAVAILABLE = 1,
    MR_ERROR_INVALID_ARGUMENT = 2,
    MR_ERROR_OUT_OF_MEMORY = 3,
    MR_ERROR_NOT_FOUND = 4,
    MR_ERROR_COMPILE = 5,
    MR_ERROR_PIPELINE = 6,
    MR_ERROR_COMMAND = 7,
    MR_ERROR_INTERNAL = 8
};

enum mr_arg_kind {
    MR_ARG_BUFFER = 1,
    MR_ARG_FLOAT32 = 2,
    MR_ARG_INT32 = 3,
    MR_ARG_UINT32 = 4,
    MR_ARG_UINT8 = 5,
    MR_ARG_FLOAT64 = 6,
    MR_ARG_INT64 = 7,
    MR_ARG_UINT64 = 8
};

typedef struct mr_grid {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
} mr_grid;

/*
 * Scalar values are stored little-endian in bits. For MR_ARG_BUFFER, buffer
 * is a shared-allocation address (interior pointers are accepted) and size is
 * the optional minimum accessible byte span.
 */
typedef struct mr_arg {
    uint32_t kind;
    uint32_t size;
    const void *buffer;
    uint64_t bits;
} mr_arg;

typedef struct mr_device_info {
    char name[256];
    uint32_t unified_memory;
    uint32_t reserved;
    uint64_t max_threads_x;
    uint64_t max_threads_y;
    uint64_t max_threads_z;
    uint64_t recommended_max_working_set_size;
    uint64_t current_allocated_size;
    uint64_t tracked_allocation_size;
    uint64_t tracked_peak_allocation_size;
} mr_device_info;

/*
 * Opaque Metal objects borrowed from the runtime. They remain valid only
 * between mr_begin_external and mr_end_external. Objective-C++ adapters may
 * bridge-cast these values to id<MTLDevice>, id<MTLCommandQueue>, and
 * id<MTLCommandBuffer>. They must append work without committing the command
 * buffer.
 */
typedef struct mr_external_context {
    void *device;
    void *queue;
    void *command_buffer;
    uint64_t token;
} mr_external_context;

typedef struct mr_buffer_view {
    void *buffer;
    size_t offset;
    size_t length;
} mr_buffer_view;

int mr_initialize(char **error_message);
int mr_shutdown(char **error_message);
int mr_get_device_info(mr_device_info *info, char **error_message);

int mr_register_source(const void *source, size_t length, char **error_message);
int mr_register_library(const void *library, size_t length, char **error_message);

int mr_alloc(size_t bytes, void **pointer, char **error_message);
int mr_free(void *pointer, char **error_message);
int mr_get_address_range(const void *pointer,
                         void **base,
                         size_t *bytes,
                         char **error_message);
int mr_copy(void *dst, const void *src, size_t bytes, char **error_message);
int mr_copy_to_device(void *dst, const void *src, size_t bytes, char **error_message);
int mr_copy_to_host(void *dst, const void *src, size_t bytes, char **error_message);
int mr_fill(void *dst, uint8_t value, size_t bytes, char **error_message);
int mr_fill_u32(void *dst, uint32_t value, size_t count, char **error_message);

int mr_launch(const char *name,
              mr_grid grid,
              const mr_arg *args,
              size_t arg_count,
              char **error_message);
int mr_flush(char **error_message);
int mr_synchronize(char **error_message);

int mr_begin_external(mr_external_context *context, char **error_message);
int mr_resolve_buffer_locked(const void *pointer,
                             size_t minimum_bytes,
                             mr_buffer_view *view,
                             char **error_message);
int mr_end_external(mr_external_context *context,
                    int encoder_failed,
                    char **error_message);

void mr_free_error(char *error_message);

#ifdef __cplusplus
}
#endif

#endif
