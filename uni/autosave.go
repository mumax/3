package uni

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/dump"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

type Autosaver struct {
	out   *dump.Writer
	data  nimble.RChanN
	every int
	hostBuf
	Dev    Device
	stream cu.Stream
}

func Autosave(data_ nimble.ChanN, every int, dev Device) {
	data := data_.NewReader()
	fname := data.Tag() + ".dump"
	r := new(Autosaver)
	r.out = dump.NewWriter(core.OpenFile(core.OD+fname), dump.CRC_ENABLED)
	data := NewReader()
	r.out.Components = data.NComp()
	r.out.MeshSize = data.Mesh().Size()
	r.out.MeshStep = data.Mesh().CellSize()
	r.data = data
	r.every = every
	r.Dev = dev
	r.stream = cu.StreamCreate()
	nimble.Stack(r)
}

func (r *Autosaver) Run() {
	N := r.data.Mesh().NCell()

	if !r.data.MemType().CPUAccess() {
		core.Assert(r.Dev != nil)
		r.Dev.InitThread()
	}

	buffer := make([][]float32, r.data.NComp())
	for c := range buffer {
		buffer[c] = make([]float32, N)
		cu.MemHostRegister(unsafe.Pointer(&buffer[c][0]), cu.SIZEOF_FLOAT32*int64(len(buffer[c])), cu.MEMHOSTREGISTER_PORTABLE)
	}

	for i := 0; ; i++ {

		if i%r.every == 0 {

			// read first
			for c := range buffer {
				data := r.data.Comp(c).ReadNext(N)
				data.Device().CopyDtoHAsync(buffer[c], r.stream)
				r.stream.Synchronize()
				r.data.ReadDone()
			}

			// then output in parallel
			i = 0
			r.out.WriteHeader()
			for c := range buffer {
				r.out.WriteData(buffer[c])
			}
			r.out.WriteHash()
		} else {
			// skip frame
			r.data.ReadNext(N)
			r.data.ReadDone()
		}
	}
}

type hostBuf []float32

func (r *hostBuf) gethost(data nimble.Slice, s cu.Stream) []float32 {
	if data.CPUAccess() {
		panic("uni.autosave cpu: need a copy")
		return data.Host()
	} // else
	if *r == nil {
		core.Debug("alloc buffer")
		*r = make([]float32, data.Len())
	}
	data.Device().CopyDtoHAsync(*r, s)
	s.Synchronize()
	return *r
}
