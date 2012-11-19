package cpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"code.google.com/p/nimble-cube/uni"
)

func NewSum(tag string, term1, term2 nimble.Chan, weight1, weight2 float32, mem nimble.MemType) *uni.Sum {
	return uni.NewSum(tag, term1, term2, weight1, weight2, mem, CPUDevice)
}
