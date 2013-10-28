package cuda

import "github.com/barnex/cuda5/cu"

var stream [3]cu.Stream // 3 general-purpose CUDA streams, one per vector component

func initStreampool() {
	for i := range stream {
		stream[i] = cu.StreamCreate()
	}
}

func SyncAll() {
	for i := range stream {
		stream[i].Synchronize()
	}
}

func Sync(str int) {
	stream[str].Synchronize()
}
