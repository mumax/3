package io

import "fmt"

// header for dump data frame
type header struct {
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

func (h *header) String() string {
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
