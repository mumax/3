// OVF2 suport written by Mykola Dvornik for mumax1,
// modified for mumax2 by Arne Vansteenkiste, 2011.
// modified for mumax3 by Arne Vansteenkiste, 2013.

package data

import (
	"fmt"
	"io"
	"log"
	"strings"
	"unsafe"
)

func DumpOvf2(out io.Writer, q *Slice, dataformat string, meta Meta) {

	switch strings.ToLower(dataformat) {
	case "binary", "binary 4":
		dataformat = "binary 4"
	case "text":
		dataformat = "text"
	default:
		log.Fatalf("Illegal OMF data format: %v", dataformat)
	}

	writeOvf2Header(out, q, meta)
	writeOvf2Data(out, q, dataformat)
	hdr(out, "End", "Segment")
	return
}

func writeOvf2Data(out io.Writer, q *Slice, dataformat string) {
	hdr(out, "Begin", "Data "+dataformat)
	switch strings.ToLower(dataformat) {
	case "text":
		writeOmfText(out, q)
	case "binary 4":
		writeOvf2Binary4(out, q)
	default:
		log.Fatalf("Illegal OMF data format: %v. Options are: Text, Binary 4", dataformat)
	}
	hdr(out, "End", "Data "+dataformat)
}

func writeOvf2Header(out io.Writer, q *Slice, meta Meta) {
	gridsize := q.Size()
	cellsize := meta.CellSize

	fmt.Fprintln(out, "# OOMMF OVF 2.0")
	fmt.Fprintln(out, "#")
	hdr(out, "Segment count", "1")
	fmt.Fprintln(out, "#")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")
	fmt.Fprintln(out, "#")

	hdr(out, "Title", meta.Name)
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")

	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)

	hdr(out, "xmax", cellsize[Z]*float64(gridsize[Z]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[X]*float64(gridsize[X]))

	name := meta.Name
	var labels []interface{}
	if q.NComp() == 1 {
		labels = []interface{}{name}
	} else {
		for i := 0; i < q.NComp(); i++ {
			labels = append(labels, name+"_"+string('x'+i))
		}
	}
	hdr(out, "valuedim", q.NComp())
	hdr(out, "valuelabels", labels...) // TODO
	unit := meta.Unit
	if unit == "" {
		unit = "1"
	}
	if q.NComp() == 1 {
		hdr(out, "valueunits", unit)
	} else {
		hdr(out, "valueunits", unit, unit, unit)
	}

	// We don't really have stages
	//fmt.Fprintln(out, "# Desc: Stage simulation time: ", meta.TimeStep, " s") // TODO
	fmt.Fprintln(out, "# Desc: Total simulation time: ", meta.Time, " s")

	hdr(out, "xbase", cellsize[Z]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[X]/2)

	hdr(out, "xnodes", gridsize[Z])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[X])

	hdr(out, "xstepsize", cellsize[Z])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[X])
	fmt.Fprintln(out, "#")
	hdr(out, "End", "Header")
	fmt.Fprintln(out, "#")
}

func writeOvf2Binary4(out io.Writer, array *Slice) {
	data := array.Tensors()
	gridsize := array.Size()

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OMF_CONTROL_NUMBER
	// Conversion form float32 [4]byte in big-endian (encoding/binary is too slow)
	bytes = (*[4]byte)(unsafe.Pointer(&controlnumber))[:]
	out.Write(bytes)

	ncomp := array.NComp()
	for ix := 0; ix < gridsize[X]; ix++ {
		for iy := 0; iy < gridsize[Y]; iy++ {
			for iz := 0; iz < gridsize[Z]; iz++ {
				for c := 0; c < ncomp; c++ {
					bytes = (*[4]byte)(unsafe.Pointer(&data[c][iz][iy][ix]))[:]
					out.Write(bytes)
				}
			}
		}
	}
}

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key string, value ...interface{}) (err error) {
	_, err = fmt.Fprint(out, "# ", key, ": ")
	_, err = fmt.Fprintln(out, value...)
	return
}

// Writes data in OMF Text format
func writeOmfText(out io.Writer, tens *Slice) (err error) {

	data := tens.Tensors()
	gridsize := tens.Size()
	ncomp := tens.NComp()

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for iz := 0; iz < gridsize[Z]; iz++ {
		for iy := 0; iy < gridsize[Y]; iy++ {
			for ix := 0; ix < gridsize[Z]; ix++ {
				for c := 0; c < ncomp; c++ {
					_, err = fmt.Fprint(out, data[c][iz][iy][ix], " ")
				}
				_, err = fmt.Fprint(out, "\n")
			}
		}
	}
	return
}

const OMF_CONTROL_NUMBER = 1234567.0 // The omf format requires the first encoded number in the binary data section to be this control number
