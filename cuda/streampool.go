package cuda

import "github.com/barnex/cuda5/cu"

const stream0 = cu.Stream(0) // for readability

func Sync() {
	stream0.Synchronize()
}
