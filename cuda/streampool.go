package cuda

// Management of a pool of re-usable CUDA streams.

import "github.com/barnex/cuda5/cu"

var streamPool chan cu.Stream // global pool of streams.
const streamPoolSize = 64     // number of streams in global pool.

// Returns a CUDA stream from a global pool.
// After use it should be recycled with SyncAndRecycle.
func stream() cu.Stream {
	return <-streamPool
}

// SyncAndRecycle takes a CUDA stream obtained by Stream(),
// the stream is synchronized and returned to the pool.
// Not recycling streams will cause deadlock quickly as
// the pool drains.
func syncAndRecycle(str cu.Stream) {
	str.Synchronize()
	streamPool <- str
}

func initStreampool() {
	streamPool = make(chan cu.Stream, streamPoolSize)
	for i := 0; i < streamPoolSize; i++ {
		streamPool <- cu.StreamCreate()
	}
}
