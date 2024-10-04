// package oommf provides the OVF data format as used by OOMMF.
package oommf

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
	"io"
	"os"
	"strconv"
	"strings"
)

// Read any OOMMF file, autodetect OVF1/OVF2 format
func Read(in io.Reader) (s *data.Slice, meta data.Meta, err error) {
	//in := fullReader{bufio.NewReader(in_)}
	info := readHeader(in)

	n := info.Size
	c := info.StepSize
	if c == [3]float64{0, 0, 0} {
		c = [3]float64{1, 1, 1} // default (presumably unitless) cell size
	}
	data_ := data.NewSlice(info.NComp, n)

	format := strings.ToLower(info.Format)
	ovf := info.OVF

	switch {
	default:
		panic(fmt.Sprint("unknown format: OVF", ovf, " ", format))
	case format == "text":
		readOVFDataText(in, data_)
	case format == "binary 4" && ovf == 1:
		readOVF1DataBinary4(in, data_)
	case format == "binary 8" && ovf == 1:
		readOVF1DataBinary8(in, data_)
	case format == "binary 4" && ovf == 2:
		readOVF2DataBinary4(in, data_)
	case format == "binary 8" && ovf == 2:
		readOVF2DataBinary8(in, data_)
	}

	return data_, data.Meta{Name: info.Title, Time: info.TotalTime, Unit: info.ValueUnit, CellSize: info.StepSize}, nil
}

func ReadFile(fname string) (*data.Slice, data.Meta, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, data.Meta{}, err
	}
	defer f.Close()
	return Read(bufio.NewReader(f))
}

func MustReadFile(fname string) (*data.Slice, data.Meta) {
	s, t, err := ReadFile(fname)
	util.FatalErr(err)
	return s, t
}

// omf.Info represents the header part of an omf file.
// TODO: add Err to return error status
// Perhaps CheckErr() func
type Info struct {
	Desc            map[string]interface{}
	Title           string
	NComp           int
	Size            [3]int
	ValueMultiplier float32
	ValueUnit       string
	Format          string // binary or text
	OVF             int
	TotalTime       float64
	StageTime       float64
	SizeofFloat     int // 4/8
	StepSize        [3]float64
	MeshUnit        string
}

// Parses the header part of the OVF1/OVF2 file
func readHeader(in io.Reader) *Info {
	desc := make(map[string]interface{})
	info := new(Info)
	info.Desc = desc

	line, eof := readLine(in)
	switch strings.ToLower(line) {
	default:
		panic("unknown header: " + line)
	case "# oommf ovf 2.0":
		info.OVF = 2
	case "# oommf: rectangular mesh v1.0":
		info.OVF = 1
		info.NComp = 3 // OVF1 only supports vector
	}
	line, eof = readLine(in)
	for !eof && !isHeaderEnd(line) {
		key, value := parseHeaderLine(line)

		switch strings.ToLower(key) {
		default:
			panic("Unknown key: " + key)
			// ignored
		case "oommf", "segment count", "begin", "meshtype", "xbase", "ybase", "zbase", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "valuerangeminmag", "valuerangemaxmag", "end": // ignored (OVF1)
		case "", "valuelabels": // ignored (OVF2)
		case "title":
			info.Title = value
		case "valueunits":
			info.ValueUnit = strings.Split(value, " ")[0] // take unit of first component, we don't support per-component units
		case "valuedim":
			info.NComp = atoi(value)
		case "xnodes":
			info.Size[X] = atoi(value)
		case "ynodes":
			info.Size[Y] = atoi(value)
		case "znodes":
			info.Size[Z] = atoi(value)
		case "xstepsize":
			info.StepSize[X] = atof(value)
		case "ystepsize":
			info.StepSize[Y] = atof(value)
		case "zstepsize":
			info.StepSize[Z] = atof(value)
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
			desc[desc_key] = desc_value
		}

		line, eof = readLine(in)
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
		info.Format = "binary " + strs[2] // binary + 4 or 8
	} else {
		info.Format = "text"
	}

	// OVF1-style time info
	if t1, ok := info.Desc["Time (s)"]; ok {
		timestr := fmt.Sprint(t1)
		t, _ := strconv.ParseFloat(timestr, 64)
		info.TotalTime = t
	}
	// OVF2-style time info
	if t2, ok := info.Desc["Total simulation time"]; ok {
		timestr := fmt.Sprint(t2)
		words := strings.Split(timestr, " ")
		t, _ := strconv.ParseFloat(words[0], 64)
		info.TotalTime = t
	}
	return info
}

// INTERNAL: Splits "# key: value" into "key", "value".
// Both may be empty
func parseHeaderLine(str string) (key, value string) {
	//remove the comment first, I *hate* go for having slices like python
	//AND not allowing negative indexes
	comPos := strings.Index(str, "##")
	if comPos != -1 {
		str = str[:comPos]
	}
	//if line doesn't begin with # just propagate it to generate proper error messages
	if !strings.HasPrefix(str, "#") {
		return str, ""
	}
	//otherwise proceed to crunch line as normal
	//TODO: check about implementing proper white space character culling instead of just looking for spaces
	strs := strings.SplitN(str, ":", 2)
	key = strings.Trim(strs[0], "# ")
	if len(strs) != 2 {
		return key, ""
	}
	value = strings.Trim(strs[1], "# ")
	return key, value
}

// INTERNAL: true if line starts with "# begin:data"
func isHeaderEnd(str string) bool {
	str = strings.ToLower(strings.Trim(str, "# "))
	str = strings.Replace(str, " ", "", -1)
	return strings.HasPrefix(str, "begin:data")
}

const OVF_CONTROL_NUMBER_4 = 1234567.0 // The omf format requires the first encoded number in the binary data section to be this control number
const OVF_CONTROL_NUMBER_8 = 123456789012345.0

// read data block in text format, for OVF1 and OVF2
func readOVFDataText(in io.Reader, t *data.Slice) {
	size := t.Size()
	data := t.Tensors()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < t.NComp(); c++ {
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
			for ix := 0; ix < gridsize[X]; ix++ {
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
	util.FatalErr(err)
	_, err = fmt.Fprintln(out, value...)
	util.FatalErr(err)
}

func dsc(out io.Writer, k, v interface{}) {
	hdr(out, "Desc", k, ": ", v)
}
