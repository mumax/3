package main

import (
	"flag"
	_ "nimble-cube/gpu"
	"nimble-cube/nimble"
	"strconv"
)

func main() {
	nimble.Check(flag.NArg() == 3, "need 3 command-line arguments: grid size")
	N0, N1, N2 := intArg(0), intArg(1), intArg(2)
	gpu.TestSymm2(N0, N1, N2)
}

func intArg(idx int) int {
	val, err := strconv.Atoi(flag.Arg(idx))
	nimble.Fatal(err)
	return val
}
