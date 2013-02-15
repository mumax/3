package kernel

//#include "reduce.h"
import "C"

// Block size for reduce kernels.
const REDUCE_BLOCKSIZE = C.REDUCE_BLOCKSIZE
