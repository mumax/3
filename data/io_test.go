package data

import (
	"bytes"
	"testing"
)

func TestIO(t *testing.T) {
	mesh := NewMesh(4, 5, 6, 1e-9, 2e-9, 3e-9)
	slice := NewSlice(3, mesh)
	data := slice.Host()
	for c := range data {
		for i := range data[c] {
			data[c][i] = float32(c * i)
		}
	}
	time := 1e-15

	buf := bytes.NewBuffer(nil)

	err := Write(buf, slice, time)
	if err != nil {
		t.Error(err)
	}

	slice2, t2, err2 := Read(buf)
	if err != nil {
		t.Error(err2)
	}

	if t2 != time {
		t.Fail()
	}

	if *slice2.Mesh() != *slice.Mesh() {
		t.Fail()
	}
	data2 := slice2.Host()
	for c := range data {
		for i := range data[c] {
			if data[c][i] != data2[c][i] {
				t.Fail()
			}
		}
	}
}
