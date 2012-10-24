package conv

import (
	"github.com/barnex/cuda5/cu"
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// Downloader for vector data specifically tuned for convolution.
// Prioritizes upload of all X components, then Y, then Z.
type Downloader struct {
	dev    [3]gpu.RChan1
	host   core.Chan3
	bsize  int
	stream cu.Stream
}

func NewDownloader(devdata [3]gpu.RChan1, hostdata core.Chan3) *Downloader {
	core.Assert(hostdata.Size() == devdata[0].Size())
	blocklen := core.Prod(core.BlockSize(hostdata.Size()))
	return &Downloader{devdata, hostdata, blocklen, 0}
}

func (u *Downloader) Run() {
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

		var in [3][]float32
		for c := 0; c < 3; c++ {
			out := u.dev[c].ReadNext(u.bsize)

			// a bit of acrobacy to lock read before write
			if c == 0 {
				in = u.host.WriteNext(u.bsize)
			}

			out.CopyDtoHAsync(in[c], u.stream)
			u.stream.Synchronize()
			u.dev[c].ReadDone()
		}
		u.host.WriteDone()
	}
}
