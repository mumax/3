package mx

import (
	"fmt"
)

func QString(q Quant) string {
	if q == nil {
		return "<nil Quant>"
	}
	return fmt.Sprintf("%#v", q)
}
