package io

import (
	"bytes"
	"code.google.com/p/mx3/mx"
	"testing"
)

func TestDump(t *testing.T) {
	mesh := mx.NewMesh(4, 5, 6, 1e-9, 2e-9, 3e-9)
	slice := mx.NewCPUSlice(3, mesh)
	data := slice.Host()
	for c := range data {
		for i := range data[c] {
			data[c][i] = float32(c * i)
		}
	}
	time := 1e-15

	buf := bytes.NewBuffer(nil)

	err := DumpSlice(buf, slice, time)
	if err != nil {
		t.Error(err)
	}

	slice2, err2 := ReadSlice(buf)
	if err != nil {
		t.Error(err2)
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
