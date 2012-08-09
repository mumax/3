package dump

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"os"
	"testing"
)

func TestDump(t *testing.T) {
	var buf_ bytes.Buffer
	buf := &buf_
	w := NewWriter(buf, CRC_DISABLED)

	w.TimeLabel = "t(s)"
	w.Time = 1e-15
	w.SpaceLabel = "r(m)"
	w.CellSize = [3]float64{1e-9, 2e-9, 3e-9}
	size := [4]int{1, 2, 4, 9}
	w.Rank = 4 // TODO: remove, detect form size
	w.Size = size[:]
	w.Precission = FLOAT32

	w.WriteHeader()
	list := make([]float32, size[0]*size[1]*size[2]*size[3])
	for i := range list {
		list[i] = float32(i)
	}
	w.WriteData(list)
	w.WriteHash()
	if w.Err != nil {
		panic(w.Err)
	}

	//buf.Bytes()[17]=42// intro error
	fmt.Println(hex.Dump(buf.Bytes()))

	r := NewReader(buf, CRC_ENABLED)
	r.Read()
	r.Fprint(os.Stdout)
}
