package oommf

import (
	"fmt"
	"github.com/mumax/3/data"
	"io"
	"log"
	"strings"
	"unsafe"
)

func WriteOVF2(out io.Writer, q *data.Slice, dataformat string, meta data.Meta) {
	writeOvf2Header(out, q, meta)
	writeOVF2Data(out, q, dataformat)
	hdr(out, "End", "Segment")
}

func writeOvf2Header(out io.Writer, q *data.Slice, meta data.Meta) {
	gridsize := q.Size()
	cellsize := meta.CellSize

	//fmt.Fprintln(out, "# OOMMF OVF 2.0")
	hdr(out, "OOMMF", "rectangular mesh v1.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")

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
	hdr(out, "Desc", "Total simulation time: ", meta.Time, " s")

	hdr(out, "xbase", cellsize[Z]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[X]/2)
	hdr(out, "xnodes", gridsize[Z])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[X])
	hdr(out, "xstepsize", cellsize[Z])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[X])
	hdr(out, "End", "Header")
}

func writeOVF2Data(out io.Writer, q *data.Slice, dataformat string) {
	hdr(out, "Begin", "Data "+dataformat)
	switch strings.ToLower(dataformat) {
	case "text":
		writeOVFText(out, q)
	case "binary 4":
		writeOVF2DataBinary4(out, q)
	default:
		log.Fatalf("Illegal OMF data format: %v. Options are: Text, Binary 4", dataformat)
	}
	hdr(out, "End", "Data "+dataformat)
}

func writeOVF2DataBinary4(out io.Writer, array *data.Slice) {
	data := array.Tensors()
	gridsize := array.Size()

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OVF_CONTROL_NUMBER_4
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

func readOVF2DataBinary4(in io.Reader, array *data.Slice) {
	size := array.Size()
	data := array.Tensors()

	var bytes4 [4]byte
	bytes := bytes4[:]
	in.Read(bytes)

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = 0.

	controlnumber = *((*float32)(unsafe.Pointer(&bytes4)))
	if controlnumber != OVF_CONTROL_NUMBER_4 {
		panic("invalid control number: " + fmt.Sprint(controlnumber))
	}

	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < 3; c++ {
					n, err := in.Read(bytes)
					if err != nil || n != 4 {
						panic(err)
					}
					data[c][iz][iy][ix] = *((*float32)(unsafe.Pointer(&bytes4)))
				}
			}
		}
	}
}
