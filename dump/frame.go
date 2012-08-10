package dump

import (
	"fmt"
	"io"
	"nimble-cube/core"
)

// Magic number
const MAGIC = "#dump100"

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
	TimeLabel  string
	Time       float64
	SpaceLabel string
	CellSize   [3]float64
	Rank       int
	Size       []int
	Precission uint64
}

func (h *Header) String() string {
	return fmt.Sprintf(
		`     magic: %v
    tlabel: %v
         t: %v
    rlabel: %v
  cellsize: %v
      rank: %v
      size: %v
precission: %v
`, h.Magic, h.TimeLabel, h.Time, h.SpaceLabel, h.CellSize, h.Rank, h.Size, h.Precission)
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

func (f *Frame) Floats() [][][]float32 {
	x := f.Tensors()
	if len(x) != 1 {
		panic(fmt.Errorf("size should be [1, x, x, x], got %v", f.Size))
	}
	return x[0]
}

func (f *Frame) Vectors() [3][][][]float32 {
	x := f.Tensors()
	if len(x) != 3 {
		panic(fmt.Errorf("size should be [3, x, x, x], got %v", f.Size))
	}
	return [3][][][]float32{x[0], x[1], x[2]}
}

func (f *Frame) Tensors() [][][][]float32 {
	if f.Rank != 4 {
		panic(fmt.Errorf("only rank 4 supported, got %v", f.Rank))
	}
	return core.Reshape4D(f.Data, f.Size)
}
