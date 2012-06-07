package main

import (
	"nimble-cube/mm"
	. "nimble-cube/nc"
)

func main() {
	// 0) initialize size, warp, etc
	InitSize(1, 4, 8)
	mm.Main()
}
