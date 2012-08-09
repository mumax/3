package dump

import (
	"bytes"
	"os"
	"testing"
)

func TestDump(t *testing.T) {
	var buf_ bytes.Buffer
	buf := &buf_
	w := NewWriter(buf, CRC_ENABLED)
	size := [3]int{4, 8, 16}
	w.Size = size[:]
	w.CellSize = [3]float64{1e-9, 2e-9, 3e-9}
	w.WriteHeader()
	list := make([]float32, size[0]*size[1]*size[2])
	w.WriteData(list)
	w.WriteHash()

	r := NewReader(buf)
	r.Read()
	r.Fprint(os.Stdout)
}
