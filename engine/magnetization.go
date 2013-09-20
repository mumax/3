package engine

type magnetization struct {
	buffered
}

func (m *magnetization) TableData() []float64 {
	avg := Average(m)
	for i := range avg {
		avg[i] /= spaceFill
	}
	return avg
}
