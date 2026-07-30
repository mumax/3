#ifndef MUMAX3_METAL_RNG_BRIDGE_H
#define MUMAX3_METAL_RNG_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mrg_status {
    MRG_SUCCESS = 0,
    MRG_ERROR_INVALID_ARGUMENT = 1,
    MRG_ERROR_RUNTIME = 2,
    MRG_ERROR_COMPILE = 3,
    MRG_ERROR_ENCODING = 4
};

void *mrg_generator_create(int32_t rng_type, char **error_message);
int mrg_generator_set_seed(void *generator, uint64_t seed, char **error_message);
int mrg_generate_normal(void *generator,
                        void *output,
                        int64_t count,
                        float mean,
                        float standard_deviation,
                        char **error_message);
int mrg_generate_raw(void *generator,
                     void *output,
                     int64_t block_count,
                     char **error_message);
void mrg_free_error(char *error_message);

#ifdef __cplusplus
}
#endif

#endif
