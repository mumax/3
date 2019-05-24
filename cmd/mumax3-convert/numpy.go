package main

import (
	"encoding/binary"
	"fmt"
	"github.com/mumax/3/data"
	"io"
)

func dumpNUMPY(f *data.Slice, info data.Meta, out io.Writer) {

	// see npy format: https://www.numpy.org/devdocs/reference/generated/numpy.lib.format.html

	// write the first 10 bytes of the 128 byte header
	fmt.Fprintf(out, "\x93NUMPY")                       // magic string
	fmt.Fprintf(out, "\x01\x00")                        // npy format version
	binary.Write(out, binary.LittleEndian, uint16(118)) // length of the actual header data (128-10)

	// write the actual header data (118 bytes)
	shapestr := fmt.Sprintf("(%d,%d,%d,%d)", f.NComp(), f.Size()[2], f.Size()[1], f.Size()[0])
	headerData := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shapestr)
	fmt.Fprintf(out, "%-117v\n", headerData) // pad with empty spaces and a newline

	// write the data
	a := f.Tensors()
	for _, a := range a {
		for _, a := range a {
			for _, a := range a {
				for i := 0; i < len(a); i++ {
					binary.Write(out, binary.LittleEndian, a[i])
				}
			}
		}
	}
}
