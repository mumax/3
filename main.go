package main

import (
	"fmt"
)

func main() {

	a := NewUniformScalar(0)
	a.name = "a"

	fmt.Println(QString(a))
}

func QString(q Quant) string {
	return fmt.Sprintf("%#v", q)
}
