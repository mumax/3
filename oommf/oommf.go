package oommf

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"io"
	"os"
)

// Read any OOMMF file, autodetect OVF1/OVF2 format
func Read(fname string) (s *data.Slice, meta data.Meta, err error) {
	in_, err := os.Open(fname)
	util.FatalErr(err)
	in := BlockingReader{bufio.NewReader(in_)}
	info := ReadHeader(in)

	n := info.Size
	c := info.StepSize
	if c == [3]float32{0, 0, 0} {
		c = [3]float32{1, 1, 1} // default (presumably unitless) cell size
	}
	data_ := data.NewSlice(3, n)

	switch info.Format {
	default:
		panic("Unknown format: " + info.Format)
	case "text":
		readOVFText(in, data_)
	case "binary":
		switch info.DataFormat {
		default:
			panic("Unknown format: " + info.Format + " " + info.DataFormat)
		case "4":
			readDataBinary4(in, data_)
		}
	}
	return data_, data.Meta{Time: info.TotalTime, Unit: info.ValueUnit}, nil
}

// omf.Info represents the header part of an omf file.
// TODO: add Err to return error status
// Perhaps CheckErr() func
type Info struct {
	Desc            map[string]interface{}
	Size            [3]int
	ValueMultiplier float32
	ValueUnit       string
	Format          string // binary or text
	OVFVersion      int
	TotalTime       float64
	StageTime       float64
	DataFormat      string // 4 or 8
	StepSize        [3]float32
	MeshUnit        string
}

const OVF_CONTROL_NUMBER_4 = 1234567.0 // The omf format requires the first encoded number in the binary data section to be this control number

// read data block in text format, for OVF1 and OVF2
func readOVFText(in io.Reader, t *data.Slice) {
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

// write data block in text format, for OVF1 and OVF2
func writeOVFText(out io.Writer, tens *data.Slice) (err error) {
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

func readLine(in io.Reader) (line string, eof bool) {
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
