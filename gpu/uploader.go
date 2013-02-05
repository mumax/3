package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
)

// Uploads data from host to GPU.
type Uploader struct {
	host nimble.RChanN
	dev  nimble.ChanN
}

// TODO: rm tag, unit
func NewUploader(tag, unit string, hostdata_ nimble.ChanN) *Uploader {
	hostdata := hostdata_.NewReader()
	panic("register")
	//MemHostRegister(hostdata_.UnsafeData().Host())
	devdata := nimble.MakeChanN(hostdata.NComp(), tag, unit, hostdata.Mesh(), nimble.GPUMemory, 0) // TODO: use same buflen as source
	u := &Uploader{hostdata, devdata}
	nimble.Stack(u)
	return u
}

func (u *Uploader) Output() nimble.ChanN {
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
			host.ReadDone()
			dev.WriteDone()
		}
	}

}
