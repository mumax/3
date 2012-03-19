package mx

import (
	"fmt"
)

// Abstract Quantity, has only name+unit.
type AQuant struct {
	name string
	unit string
}

func (this *AQuant) Name() string {
	if this.name != "" {
		return this.name
	}
	return fmt.Sprintf("Q%p", &this) // default name if none set: Q0x000abcd
}

func (this *AQuant) Unit() string {
	return this.unit
}
