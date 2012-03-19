package mx

import (
	"fmt"
)

func QString(q Quant) string { //←[ leaking param: q]
	if q == nil {
		return "Q<nil>"
	}
	return fmt.Sprintf("%s: %#v", q.Name(), q) //←[ QString ... argument does not escape]
}
