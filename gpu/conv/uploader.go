package conv

import (
	"github.com/barnex/cuda5/cu"
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// Uploader for vector data specifically tuned for convolution.
// Prioritizes upload of all X components, then Y, then Z.
type Uploader struct {
	host   core.RChan3
	dev    [3]gpu.Chan1
	bsize  int
	stream cu.Stream
}

func NewUploader(hostdata core.RChan3, devdata [3]gpu.Chan1) *Uploader {
	core.Assert(hostdata.Mesh().Size() == devdata[0].Size())
	blocklen := core.Prod(core.BlockSize(hostdata.Mesh().Size()))
	return &Uploader{hostdata, devdata, blocklen, 0}
}

func (u *Uploader) Run() {
	core.Debug("run conv.uploader with block size", u.bsize)
	gpu.LockCudaThread()
	defer gpu.UnlockCudaThread()
	u.stream = cu.StreamCreate()
	for c := 0; c < 3; c++ {
		gpu.MemHostRegister(u.host.UnsafeData()[c])
	}

	for {
		// -- here be dragons
		// TODO: properly prioritized implementation
		in := u.host.ReadNext(u.bsize)
		for c := 0; c < 3; c++ {
			out := u.dev[c].WriteNext(u.bsize)
			//core.Debug("upload", c, u.bsize)
			out.CopyHtoDAsync(in[c], u.stream)
			u.stream.Synchronize()
			u.dev[c].WriteDone()
		}
		u.host.ReadDone()
	}
}
