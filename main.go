package main

import (
	. "fe-mumax/mx"
	"fmt"
)

func main() {

	a := NewUniformScalar(0)
	fmt.Println(QString(a))
}

func QString(q Quant) string {
	return fmt.Sprintf("%#v", q)
}
