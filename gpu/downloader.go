package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
)

// Downloads data from GPU to host.
type Downloader struct {
	dev    nimble.RChan1
	host   nimble.Chan1
	bsize  int
	stream cu.Stream
}

func NewDownloader(devdata nimble.RChan1, hostdata nimble.Chan1) *Downloader {
	nimble.Assert(hostdata.Size() == devdata.Size())
	blocklen := nimble.Prod(nimble.BlockSize(hostdata.Size()))
	return &Downloader{devdata, hostdata, blocklen, 0} // TODO: block size
}

func (u *Downloader) Run() {
	nimble.Debug("run gpu.downloader with block size", u.bsize)
	LockCudaThread()
	u.stream = cu.StreamCreate()
	//MemHostRegister(u.host.UnsafeData()) // TODO

	for {
		in := u.dev.ReadNext(u.bsize).Device()
		out := u.host.WriteNext(u.bsize).Host()
		in.CopyDtoHAsync(out, u.stream)
		u.stream.Synchronize()
		u.host.WriteDone()
		u.dev.ReadDone()
	}
}

func RunDownloader(tag string, input nimble.Chan) nimble.ChanN {
	in := input.ChanN()
	output := nimble.MakeChanN(in.NComp(), tag, in.Unit(), in.Mesh(), nimble.CPUMemory)
	for i := 0; i < in.NComp(); i++ {
		nimble.Stack(NewDownloader(in.Comp(i).NewReader(), output.Comp(i)))
	}
	return output
}
