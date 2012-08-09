package dump

import (
	"fmt"
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
	Data []float32
	CRC  uint64
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
