package script

import "reflect"

func typecheck(l, r reflect.Type) {
	if l != r {
		panic(err("type mismatch:", l, r))
	}
}

var (
	float64_t = reflect.TypeOf(float64(0))
)
