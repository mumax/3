package nc

import (
	"fmt"
)

type Vector [3]float32

const(
	X=0
	Y=1
	Z=2
)

func(v*Vector)String()  string{
	return fmt.Sprint(v[X], ", ", v[Y], ",", v[Z])
}
