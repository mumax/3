package cpu

import (
	"code.google.com/p/mx3/nimble"
	"code.google.com/p/mx3/uni"
)

func NewSum(tag string, term1, term2 nimble.Chan, weight1, weight2 float32, mem nimble.MemType) *uni.Sum {
	return uni.NewSum(tag, term1, term2, weight1, weight2, mem, CPUDevice)
}
