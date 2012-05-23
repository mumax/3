package nc

import(
	"testing"
)

func BenchmarkString(bench *testing.B){
	v:=Vector{1,2,3}
	for i:=0;i<bench.N; i++{
		v.String()
	}
}
