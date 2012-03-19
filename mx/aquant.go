package mx

import (
	"fmt"
)

// Abstract Quantity, has only name+unit.
type AQuant struct {
	name string
	unit string
}

func (this *AQuant) Name() string { //←[ moved to heap: this]
	if this.name != "" {
		return this.name
	}
	return fmt.Sprintf("Q%p", &this) // default name if none set: Q0x000abcd//←[ &this escapes to heap  (*AQuant).Name ... argument does not escape]
}

func (this *AQuant) Unit() string { //←[ can inline (*AQuant).Unit  (*AQuant).Unit this does not escape]
	return this.unit
}

// Empty implementation
func (this *AQuant) Update() { //←[ can inline (*AQuant).Update  (*AQuant).Update this does not escape]

}
