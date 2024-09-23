//+build ignore

/*
   Test Shifted Quantity

   Shifted(Quantity q, dx, dy, dz int) shifts the input Quantity over (dx, dy, dz) cells.
   The shift is performed on the gpu when the Shifted quantity is being evaluated.
   This test checks if the shift is correctly performed.
*/

package main

import (
	"github.com/mumax/3/data"
	. "github.com/mumax/3/engine"
	"os"
)

func main() {
	defer InitAndClose()()

	// arbitrarily chosen grid
	SetGridSize(32, 16, 8)
	SetCellSize(1.7, 3.2, 7)

	// arbitrarily chosen shift
	dx, dy, dz := -4, 3, 1

	// arbitrarily created quantity q
	q := FunctionQuantity{func(r data.Vector) float64 { return 3*r.X() - r.Y() + r.Z()*r.Z() }}

	// Evaluate Shifted(q, dx, dy, dz) and copy the result to the host
	shiftedOnGpu := func() *data.Slice {
		r := ValueOf(Shifted(q, dx, dy, dz))
		defer r.Free()
		return r.HostCopy()
	}

	// Evaluate quantity q and shift the output slice on the host
	shiftedOnHost := func() *data.Slice {
		v := ValueOf(q)
		defer v.Free()
		return shiftSlice(v.HostCopy(), dx, dy, dz)
	}

	// Check if both approaches yield the same result
	if !slicesAreEqual(shiftedOnGpu(), shiftedOnHost()) {
		LogErr("Shifted(Quantity, dx, dy, dz) did not shift the input quantity correctly")
		os.Exit(1)
	}
}

// Shift slice values on the host over (dx, dy, dz) cellS
func shiftSlice(input *data.Slice, dx, dy, dz int) *data.Slice {
	if !input.CPUAccess() {
		input = input.HostCopy()
	}

	size := input.Size()
	output := data.NewSlice(1, size)

	for x := 0; x < size[X]; x++ {
		for y := 0; y < size[Y]; y++ {
			for z := 0; z < size[Z]; z++ {
				val := 0.0
				if x-dx >= 0 && x-dx < size[X] && y-dy >= 0 && y-dy < size[Y] && z-dz >= 0 && z-dz < size[Z] {
					val = input.Get(0, x-dx, y-dy, z-dz)
				}
				output.Set(0, x, y, z, val)
			}
		}
	}
	return output
}

// Return true if the values of two slices are equal to each other
func slicesAreEqual(aSlice, bSlice *data.Slice) bool {
	size := aSlice.Size()
	ncomp := aSlice.NComp()

	if bSlice.NComp() != ncomp || bSlice.Size()[X] != size[X] || bSlice.Size()[Y] != size[Y] || bSlice.Size()[Z] != size[Z] {
		return false
	}

	if !aSlice.CPUAccess() {
		aSlice = aSlice.HostCopy()
	}

	if !bSlice.CPUAccess() {
		bSlice = bSlice.HostCopy()
	}

	for x := 0; x < size[X]; x++ {
		for y := 0; y < size[Y]; y++ {
			for z := 0; z < size[Z]; z++ {
				for c := 0; c < aSlice.NComp(); c++ {
					if aSlice.Get(c, x, y, z) != bSlice.Get(c, x, y, z) {
						return false
					}
				}
			}
		}
	}

	return true
}

// Implements a (scalar) Quantity which evaluates a function on the global mesh
type FunctionQuantity struct {
	function func(data.Vector) float64
}

func (q FunctionQuantity) NComp() int {
	return 1
}

func (q FunctionQuantity) EvalTo(dst *data.Slice) {
	result := data.NewSlice(q.NComp(), dst.Size())
	for x := 0; x < result.Size()[X]; x++ {
		for y := 0; y < result.Size()[Y]; y++ {
			for z := 0; z < result.Size()[Z]; z++ {
				r := Index2Coord(x, y, z)
				result.Set(0, x, y, z, q.function(r))
			}
		}
	}
	data.Copy(dst, result)
}
