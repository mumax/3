package oommf

import (
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"io"
	"log"
	"strconv"
	"strings"
	"unsafe"
)

// Safe way to get Desc values: panics when key not present
func (i *Info) DescGet(key string) interface{} {
	value, ok := i.Desc[key]
	if !ok {
		panic("Key not found in Desc: " + key)
	}
	return value
}

// Safe way to get a float from Desc
func (i *Info) DescGetFloat32(key string) float32 {
	value := i.DescGet(key)
	fl, err := strconv.ParseFloat(value.(string), 32)
	if err != nil {
		panic("Could not parse " + key + " to float32: " + err.Error())
	}
	return float32(fl)
}

func readDataText(in io.Reader, t *data.Slice) {
	size := t.Size()
	data := t.Tensors()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < 3; c++ {
					_, err := fmt.Fscan(in, &data[c][iz][iy][ix])
					if err != nil {
						panic(err)
					}
				}
			}
		}
	}
}

func readDataBinary4(in io.Reader, t *data.Slice) {
	size := t.Size()
	data := t.Tensors()

	var bytes4 [4]byte
	bytes := bytes4[:]

	in.Read(bytes)                                                                  // TODO: must read 4 !
	bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0] // swap endianess

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = 0.

	// Conversion form float32 [4]byte, encoding/binary is too slow
	// Inlined for performance, terabytes of data will pass here...
	controlnumber = *((*float32)(unsafe.Pointer(&bytes4)))
	if controlnumber != OMF_CONTROL_NUMBER {
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
					bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0] // swap endianess
					data[c][iz][iy][ix] = *((*float32)(unsafe.Pointer(&bytes4)))
				}
			}
		}
	}

}

// INTERNAL: Splits "# key: value" into "key", "value".
// Both may be empty
func parseHeaderLine(str string) (key, value string) {
	strs := strings.SplitN(str, ":", 2)
	key = strings.Trim(strs[0], "# ")
	if len(strs) != 2 {
		return key, ""
	}
	value = strings.Trim(strs[1], "# ")
	return key, value
}

// INTERNAL: true if line == "# begin_data"
func isHeaderEnd(str string) bool {
	str = strings.ToLower(strings.Trim(str, "# "))
	str = strings.Replace(str, " ", "", -1)
	return strings.HasPrefix(str, "begin:data")
}

// Parses the header part of the omf file
func ReadHeader(in io.Reader) *Info {
	desc := make(map[string]interface{})
	info := new(Info)
	info.Desc = desc

	line, eof := ReadLine(in)
	for !eof && !isHeaderEnd(line) {
		key, value := parseHeaderLine(line)

		switch strings.ToLower(key) {
		default:
			panic("Unknown key: " + key)
			// ignored
		case "oommf", "segment count", "begin", "title", "meshtype", "xbase", "ybase", "zbase", "xstepsize", "ystepsize", "zstepsize", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "valuerangeminmag", "valuerangemaxmag", "end":
		case "xnodes":
			info.Size[X] = atoi(value)
		case "ynodes":
			info.Size[Y] = atoi(value)
		case "znodes":
			info.Size[Z] = atoi(value)
		case "valuemultiplier":
		case "valueunit":
		case "meshunit":
			// desc tags: parse further and add to metadata table
		case "desc":
			strs := strings.SplitN(value, ":", 2)
			desc_key := strings.Trim(strs[0], "# ")
			// Desc tag does not neccesarily have a key:value layout.
			// If not, we use an empty value string.
			desc_value := ""
			if len(strs) > 1 {
				desc_value = strings.Trim(strs[1], "# ")
			}
			// 			fmt.Println(desc_key, " : ", desc_value)
			desc[desc_key] = desc_value
		}

		line, eof = ReadLine(in)
	}
	// the remaining line should now be the begin:data clause
	key, value := parseHeaderLine(line)
	value = strings.TrimSpace(value)
	strs := strings.SplitN(value, " ", 3)
	if strings.ToLower(key) != "begin" || strings.ToLower(strs[0]) != "data" {
		panic("Expected: Begin: Data")
	}
	info.Format = strings.ToLower(strs[1])
	if len(strs) >= 3 { // dataformat for text is empty
		info.DataFormat = strs[2]
	}
	return info
}

func atoi(a string) int {
	i, err := strconv.Atoi(a)
	if err != nil {
		panic(err)
	}
	return i
}

// Blocks until all requested bytes are read.
type BlockingReader struct{ io.Reader }

func (r BlockingReader) Read(p []byte) (n int, err error) {
	return io.ReadFull(r.Reader, p)
}

// Reads one character from the Reader.
// -1 means EOF.
// Errors are cought and cause panic
func ReadChar(in io.Reader) int {
	buffer := [1]byte{}
	switch nr, err := in.Read(buffer[0:]); true {
	case nr < 0: // error
		panic(err)
	case nr == 0: // eof
		return -1
	case nr > 0: // ok
		return int(buffer[0])
	}
	panic("unreachable")
}

//
func ReadLine(in io.Reader) (line string, eof bool) {
	char := ReadChar(in)
	eof = isEOF(char)

	for !isEndline(char) {
		line += string(byte(char))
		char = ReadChar(in)
	}
	return line, eof
}

func isEOF(char int) bool {
	return char == -1
}

func isEndline(char int) bool {
	return isEOF(char) || char == int('\n')
}

// OVF2 suport written by Mykola Dvornik for mumax1,
// modified for mumax2 by Arne Vansteenkiste, 2011.
// modified for mumax3 by Arne Vansteenkiste, 2013, 2014

func DumpOvf2(out io.Writer, q *data.Slice, dataformat string, meta data.Meta) {
	writeOvf2Header(out, q, meta)
	writeOvf2Data(out, q, dataformat)
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

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key string, value ...interface{}) {
	_, err := fmt.Fprint(out, "# ", key, ": ")
	util.FatalErr(err, "while reading OOMMF header")
	_, err = fmt.Fprintln(out, value)
	util.FatalErr(err, "while reading OOMMF header")
}

func dsc(out io.Writer, k, v interface{}) {
	hdr(out, "Desc", k, ": ", v)
}

func writeOvf2Data(out io.Writer, q *data.Slice, dataformat string) {
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

func writeOvf2Binary4(out io.Writer, array *data.Slice) {
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

// Writes data in OMF Text format
func writeOmfText(out io.Writer, tens *data.Slice) (err error) {

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
