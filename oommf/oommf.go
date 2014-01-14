package oommf

import (
	"bufio"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
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
		readDataText(in, data_)
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
