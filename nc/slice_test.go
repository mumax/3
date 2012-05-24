package nc

import (
	"fmt"
	//"testing"
)

func ExampleSlice() {
	N := 100
	vec := MakeVectorSlice(N)
	fmt.Println("vec.N():", vec.N(), "len(vec):", len(vec))

	y := vec.VectorComp(Y)
	fmt.Println("y.N():", y.N(), "len(y):", len(y))

	// Output: 
	// vec.N(): 100 len(vec): 300
	// y.N(): 100 len(y): 100
}
