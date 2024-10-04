package oommf

import (
	"encoding/binary"
	"fmt"
	"github.com/mumax/3/v3/data"
	"io"
	"log"
	"strings"
	"unsafe"
)

func WriteOVF1(out io.Writer, q *data.Slice, meta data.Meta, dataformat string) {
	if q.NComp() != 3 {
		log.Fatal("Cannot save the quantity: the OVF1 format only supports 3D-vector fields.")
	}
	writeOVF1Header(out, q, meta)
	writeOVF1Data(out, q, dataformat)
	hdr(out, "End", "Segment")
}

func writeOVF1Data(out io.Writer, q *data.Slice, dataformat string) {
	canonicalFormat := ""
	switch strings.ToLower(dataformat) {
	case "text":
		canonicalFormat = "Text"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVFText(out, q)
	case "binary", "binary 4":
		canonicalFormat = "Binary 4"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVF1Binary4(out, q)
	default:
		log.Fatalf("Illegal OVF data format: %v. Options are: Text, Binary 4", dataformat)
	}
	hdr(out, "End", "Data "+canonicalFormat)
}

// Writes the OMF header
func writeOVF1Header(out io.Writer, q *data.Slice, meta data.Meta) {
	gridsize := q.Size()
	cellsize := meta.CellSize

	hdr(out, "OOMMF", "rectangular mesh v1.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")

	hdr(out, "Begin", "Header")

	dsc(out, "Time (s)", meta.Time)
	hdr(out, "Title", meta.Name)
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")
	hdr(out, "xbase", cellsize[X]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[Z]/2)
	hdr(out, "xstepsize", cellsize[X])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[Z])
	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)
	hdr(out, "xmax", cellsize[X]*float64(gridsize[X]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[Z]*float64(gridsize[Z]))
	hdr(out, "xnodes", gridsize[X])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[Z])
	hdr(out, "ValueRangeMinMag", 1e-08) // not so "optional" as the OOMMF manual suggests...
	hdr(out, "ValueRangeMaxMag", 1)     // TODO
	hdr(out, "valueunit", meta.Unit)
	hdr(out, "valuemultiplier", 1)

	hdr(out, "End", "Header")
}

// Writes data in OMF Binary 4 format
func writeOVF1Binary4(out io.Writer, array *data.Slice) (err error) {
	data := array.Tensors()
	gridsize := array.Size()

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OVF_CONTROL_NUMBER_4
	// Conversion form float32 [4]byte in big-endian
	// Inlined for performance, terabytes of data will pass here...
	bytes = (*[4]byte)(unsafe.Pointer(&controlnumber))[:]
	bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0] // swap endianess
	_, err = out.Write(bytes)

	ncomp := array.NComp()
	for iz := 0; iz < gridsize[Z]; iz++ {
		for iy := 0; iy < gridsize[Y]; iy++ {
			for ix := 0; ix < gridsize[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					// dirty conversion from float32 to [4]byte
					bytes = (*[4]byte)(unsafe.Pointer(&data[c][iz][iy][ix]))[:]
					bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0]
					out.Write(bytes)
				}
			}
		}
	}
	return
}

func readOVF1DataBinary4(in io.Reader, t *data.Slice) {
	size := t.Size()
	data := t.Tensors()

	// OOMMF requires this number to be first to check the format
	var controlnumber float32
	// OVF 1.0 is network byte order (MSB)
	binary.Read(in, binary.BigEndian, &controlnumber)
	if controlnumber != OVF_CONTROL_NUMBER_4 {
		panic("invalid OVF1 control number: " + fmt.Sprint(controlnumber))
	}

	var tmp float32
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < 3; c++ {
					err := binary.Read(in, binary.BigEndian, &tmp)
					if err != nil {
						panic(err)
					}
					data[c][iz][iy][ix] = tmp
				}
			}
		}
	}
}

func readOVF1DataBinary8(in io.Reader, t *data.Slice) {
	size := t.Size()
	data := t.Tensors()

	// OOMMF requires this number to be first to check the format
	var controlnumber float64
	// OVF 1.0 is network byte order (MSB)
	binary.Read(in, binary.BigEndian, &controlnumber)

	if controlnumber != OVF_CONTROL_NUMBER_8 {
		panic("invalid OVF1 control number: " + fmt.Sprint(controlnumber))
	}

	var tmp float64
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < 3; c++ {
					err := binary.Read(in, binary.BigEndian, &tmp)
					if err != nil {
						panic(err)
					}
					data[c][iz][iy][ix] = float32(tmp)
				}
			}
		}
	}
}
