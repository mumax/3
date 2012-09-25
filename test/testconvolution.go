package main

import (
	"flag"
	"nimble-cube/core"
	_ "nimble-cube/gpu"
	"nimble-cube/gpu/conv"
	"strconv"
)

func main() {
	core.Check(flag.NArg() == 3, "need 3 command-line arguments: grid size")
	N0, N1, N2 := intArg(0), intArg(1), intArg(2)
	conv.TestSymm2(N0, N1, N2)
}

func intArg(idx int) int {
	val, err := strconv.Atoi(flag.Arg(idx))
	core.Fatal(err)
	return val
}
