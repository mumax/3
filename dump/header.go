package dump

const MAGIC = "#dump10\n"

type Header struct {
	TimeLabel  string
	Time       float64
	SpaceLabel string
	CellSize   [3]float64
	Rank       int
	Size       []int
	Precission int64
}
