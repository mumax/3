package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

// Downloads data from host to GPU.
type Downloader struct {
	host   nimble.Chan1
	dev    nimble.RChan1
	stream cu.Stream
}

func NewDownloader(tag, unit string, devdata_ nimble.Chan1) *Downloader {
	devdata := devdata_.NewReader()
	hostdata := nimble.MakeChan1(tag, unit, devdata.Mesh, nimble.CPUMemory, devdata_.NBufferedBlocks())
	MemHostRegister(hostdata.UnsafeData().Host())
	u := &Downloader{hostdata, devdata, cu.StreamCreate()}
	nimble.Stack(u)
	return u
}

func (u *Downloader) Output() nimble.Chan1 {
	return u.host
}

func (u *Downloader) Run() {
	core.Debug("run gpu.downloader")
	N := u.host.BlockLen()
	LockCudaThread()

	for {
		in := u.dev.ReadNext(N).Device()
		out := u.host.WriteNext(N).Host()
		in.CopyDtoHAsync(out, u.stream)
		u.stream.Synchronize()
		u.host.WriteDone()
		u.dev.ReadDone()
	}
}

//func RunUploader(tag string, input nimble.Chan) nimble.ChanN {
//	in := input.ChanN()
//
//	output := nimble.MakeChanN(in.NComp(), tag, in.Unit(), in.Mesh(), nimble.GPUMemory, in.NBufferedBlocks())
//
//	for i := 0; i < output.NComp(); i++ {
//		nimble.Stack(NewUploader(in.Comp(i).NewReader(), output.Comp(i)))
//	}
//	return output
//}

//import (
//	"code.google.com/p/nimble-cube/core"
//	"code.google.com/p/nimble-cube/nimble"
//	"github.com/barnex/cuda5/cu"
//)
//
//// Downloads data from GPU to host.
//type Downloader struct {
//	dev    nimble.RChan1
//	host   nimble.Chan1
//	bsize  int
//	stream cu.Stream
//}
//
//// TODO: make Output()
//func NewDownloader(devdata nimble.RChan1) *Downloader {
//	core.Assert(hostdata.Size() == devdata.Size())
//	blocklen := devdata.BlockLen()
//	return &Downloader{devdata, hostdata, blocklen, 0} // TODO: block size
//}
//
//func (u *Downloader) Run() {
//	core.Debug("run gpu.downloader with block size", u.bsize)
//	LockCudaThread()
//	u.stream = cu.StreamCreate()
//	//MemHostRegister(u.host.UnsafeData()) // TODO
//
//	for {
//		in := u.dev.ReadNext(u.bsize).Device()
//		out := u.host.WriteNext(u.bsize).Host()
//		in.CopyDtoHAsync(out, u.stream)
//		u.stream.Synchronize()
//		u.host.WriteDone()
//		u.dev.ReadDone()
//	}
//}
//
//func RunDownloader(tag string, input nimble.Chan) nimble.ChanN {
//	in := input.ChanN()
//	output := nimble.MakeChanN(in.NComp(), tag, in.Unit(), in.Mesh(), nimble.CPUMemory, in.NBufferedBlocks())
//	for i := 0; i < in.NComp(); i++ {
//		nimble.Stack(NewDownloader(in.Comp(i).NewReader(), output.Comp(i)))
//	}
//	return output
//}
