package engine

import (
	"github.com/mumax/3/util"
	"math"
	"math/rand"
)

func init() {
	DeclFunc("ext_MakeGrains", makeGrains, "Make regions like randomly distributed grains (startregion, maxregion[excl.], numberofgrains)")
}

func makeGrains(startregion, stopRegion, numberofgrains int) {
	numberofregions := stopRegion - startregion
	//log.Printf("making grains in %v regions, starting from %v, number of Voronoi centra=%v", numberofregions, startregion, numberofgrains)
	util.Argument(len(regions.arr) == 1) // 2D only

	for i := startregion; i < startregion+numberofregions; i++ {
		DefRegion(i, func(x, y, z float64) bool { return false })
	}
	arr := regions.arr[0]

	//make voronoi:
	voronoi := make([][]int, len(arr))
	for i := range voronoi {
		voronoi[i] = make([]int, len(arr[i]))
	}

	// choose "numberofgrains" cells random in whole geometry en and give them voronoinumber
	//	store the voronoicenters in different slice, to be used in next step
	voronoicentra := make([][]int, numberofgrains)
	for i := range voronoicentra {
		voronoicentra[i] = make([]int, 2)
	}

	for voronoinumber := 0; voronoinumber < numberofgrains; voronoinumber++ {
		j := rand.Intn(len(arr))
		k := rand.Intn(len(arr[j]))
		voronoicentra[voronoinumber][0] = j
		voronoicentra[voronoinumber][1] = k
		voronoi[j][k] = voronoinumber
	}

	//2: Find for every cell the closest voronoicentre and change its voronoinumber to the number of the centre

	for j := 0; j < len(arr); j++ {
		for k := 0; k < len(arr[j]); k++ {
			distance := float64(math.Sqrt(math.Pow(float64(j-voronoicentra[0][0]), 2.) + math.Pow(float64(k-voronoicentra[0][1]), 2.)))
			newdist := float64(0)
			for i := range voronoicentra {
				newdist = float64(math.Sqrt(math.Pow(float64(j-voronoicentra[i][0]), 2.) + math.Pow(float64(k-voronoicentra[i][1]), 2.)))
				if newdist < distance {
					voronoi[j][k] = i
					distance = newdist
				}
			}
		}
	}

	//3: give every cell its region by diividing its voronoinumber modulo number of regions
	A := regions.arr
	for _, arr := range A {
		for j := 0; j < len(arr); j++ {
			for k := 0; k < len(arr[j]); k++ {
				arr[j][k] = byte(startregion + voronoi[j][k]%numberofregions)
			}
		}
	}
}
