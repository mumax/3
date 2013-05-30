package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

type GPU_Getter interface {
	GetGPU() (q *data.Slice, recycle bool) // get quantity data (GPU), indicate need to recycle
}

type Getter interface {
	Get() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
}

type Saver interface {
	Getter
	autoFname() string // file name for autosave
}

// Download a quantity to host,
// or just return its data when already on host.
func Download(q Getter) *data.Slice {
	buf, recycle := q.Get()
	if buf.CPUAccess() {
		util.Assert(recycle == false)
		return buf
	}
	h := hostBuf(buf.NComp(), buf.Mesh())
	data.Copy(h, buf)
	if recycle {
		cuda.RecycleBuffer(buf)
	}
	return h
}

func hostBuf(nComp int, m *data.Mesh) *data.Slice {
	return data.NewSlice(nComp, m) // TODO use pool of page-locked buffers
}

func Save(q Saver) {
	SaveAs(q, q.autoFname())
}

func SaveAs(q Getter, fname string) {
	data.MustWriteFile(fname, Download(q), Time) // async would be nice
}
