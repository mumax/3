package cuda

import "unsafe"

type LUTPtr unsafe.Pointer    // points to 256 float32's
type LUTPtrs []unsafe.Pointer // elements point to 256 float32's
type SymmLUT unsafe.Pointer   // points to 256x256 symmetric matrix, only lower half stored
