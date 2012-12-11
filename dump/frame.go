package dump

import (
	"code.google.com/p/mx3/core"
	"fmt"
	"io"
	"os"
)

// Magic number
const MAGIC = "#dump002"

// Precision identifier
const (
	FLOAT32 = 4
)

// Header+data frame.
type Frame struct {
	Header
	Data  []float32
	CRC   uint64
	Bytes int64 // Total number of bytes read.
	Err   error // Stores the latest I/O error, if any.
}

// Header for dump data frame
type Header struct {
	Magic      string
	Components int
	MeshSize   [3]int
	MeshStep   [3]float64
	MeshUnit   string
	Time       float64
	TimeUnit   string
	DataLabel  string
	DataUnit   string
	Precission uint64
}

func (h *Header) NComp() int { return h.Components }

func (h *Header) size() []int {
	return []int{h.Components, h.MeshSize[0], h.MeshSize[1], h.MeshSize[2]}
}

func (h *Header) String() string {
	return fmt.Sprintf(
		`     Magic: %v
Components: %v
  MeshSize: %v
  MeshStep: %v
  MeshUnit: %v
      Time: %v
  TimeUnit: %v
 DataLabel: %v
  DataUnit: %v
Precission: %v
`, h.Magic, h.Components, h.MeshSize, h.MeshStep, h.MeshUnit, h.Time, h.TimeUnit, h.DataLabel, h.DataUnit, h.Precission)
}

// Print the frame in human readable form.
func (f *Frame) Fprintf(out io.Writer, format string) {
	if f.Err != nil {
		fmt.Fprintln(out, f.Err)
		return
	}
	fmt.Fprintln(out, f.Header.String())
	core.Fprintf(out, format, f.Tensors())
	fmt.Fprintf(out, "ISO CRC64:%x\n", f.CRC)
}

// Print the frame in human readable form to stdout using default formatting.
func (f *Frame) Print() {
	f.Fprintf(os.Stdout, "%v")
}

func (f *Frame) Floats() [][][]float32 {
	x := f.Tensors()
	if len(x) != 1 {
		panic(fmt.Errorf("expecting 1 component, got %v", f.Components))
	}
	return x[0]
}

func (f *Frame) Vectors() [3][][][]float32 {
	x := f.Tensors()
	if len(x) != 3 {
		panic(fmt.Errorf("expecting 3 components, got %v", f.Components))
	}
	return [3][][][]float32{x[0], x[1], x[2]}
}

func (f *Frame) Tensors() [][][][]float32 {
	return core.Reshape4D(f.Data, f.size())
}
