package gpu

import "github.com/barnex/cuda4/safe"

func MakeVectors(n int) [3]safe.Float32s {
	return [3]safe.Float32s{safe.MakeFloat32s(n), safe.MakeFloat32s(n), safe.MakeFloat32s(n)}
}
