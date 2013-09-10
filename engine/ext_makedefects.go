package engine

import (
	"code.google.com/p/mx3/util"
	"log"
	"math/rand"
)

func init() {
	DeclFunc("ext_MakeDefects", makeDefects, "Make randomly distributed defects (region, size, probabilty)")
}

func makeDefects(region, size int, prob float64) {
	log.Printf("making defects region=%v, size=%vx%v, probabiltiy=%v", region, size, size, prob)
	util.Argument(len(regions.arr) == 1) // 2D only
	// make sure this region is defined (!)
	DefRegion(region, func(x, y, z float64) bool { return false })

	arr := regions.arr[0]
	r := byte(region)

	for j := 0; j < len(arr)-size; j++ {
		for k := 0; k < len(arr[j])-size; k++ {
			if rand.Float64() < prob {

				for j_ := j; j_ < j+size && j_ < len(arr); j_++ {
					for k_ := k; k_ < k+size && k_ < len(arr[j]); k_++ {
						arr[j_][k_] = r
					}
				}
			}
		}
	}
}
