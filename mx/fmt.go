package mx

import (
	"fmt"
)

func QString(q Quant) string {
	if q == nil {
		return "Q<nil>"
	}
	return fmt.Sprintf("%s: %#v", q.Name(), q)
}
