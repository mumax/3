package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"path"
	"strings"
)

type Getter interface {
	Get() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
}

func SaveAs(q Getter, fname string) {
	if !path.IsAbs(fname) && !strings.HasPrefix(fname, OD) {
		fname = path.Clean(OD + "/" + fname)
	}
	if path.Ext(fname) == "" {
		fname += ".dump"
	}
	buffer, recylce := q.Get()
	if recylce {
		defer cuda.RecycleBuffer(buffer)
	}
	AsyncSave(fname, assureCPU(buffer), Time)
}

// Copy to CPU, if needed
func assureCPU(s *data.Slice) *data.Slice {
	if s.CPUAccess() {
		return s
	} else {
		return s.HostCopy()
	}
}

// Download a quantity to host,
// or just return its data when already on host.
func Download(q Getter) *data.Slice {
	buf, recycle := q.Get()
	if recycle {
		defer cuda.RecycleBuffer(buf)
	}
	if buf.CPUAccess() {
		return buf
	} else {
		return buf.HostCopy()
	}
}
