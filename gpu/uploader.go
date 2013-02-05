package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

// Uploads data from host to GPU.
type Uploader struct {
	host nimble.RChanN
	dev  nimble.ChanN
}

// TODO: rm tag, unit
func NewUploader(tag, unit string, hostdata_ nimble.ChanN) *Uploader {
	hostdata := hostdata_.NewReader()
	MemHostRegister(hostdata_.UnsafeData().Host())
	devdata := nimble.MakeChanN(hostdata.NComp(), tag, unit, hostdata.Mesh(), nimble.GPUMemory, hostdata_.NBufferedBlocks())
	u := &Uploader{hostdata, devdata, cu.StreamCreate()}
	nimble.Stack(u)
	return u
}

func (u *Uploader) Output() nimble.Chan1 {
	return u.dev
}

func (u *Uploader) Run() {
	//	core.Debug("run gpu.uploader")
	//	N := u.host.BlockLen()
	//	LockCudaThread()
	//
	//	for {
	//		in := u.host.ReadNext(N).Host()
	//		out := u.dev.WriteNext(N).Device()
	//		out.CopyHtoDAsync(in, u.stream)
	//		u.stream.Synchronize()
	//		u.dev.WriteDone()
	//		u.host.ReadDone()
	//	}
	//
	core.Debug("run gpu.uploader")
	N := u.host.BufLen()
	panic("lock")
	//LockCudaThread()

	for {
		for c := 0; c < u.dev.NComp(); c++ {
			dev := u.dev.Comp(c)
			host := u.host.Comp(c)
			out := dev.WriteNext(N)
			in := host.ReadNext(N)
			Copy(out, in)
			host.WriteDone()
			dev.ReadDone()
		}
	}

}
