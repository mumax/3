package script

import (
	"fmt"
	"go/token"
	"reflect"
)

// compileErr, and only compileErr will be caught by Compile and returned as an error.
type compileErr struct {
	pos token.Pos
	msg string
}

// implements error
func (c *compileErr) Error() string {
	return c.msg
}

// constructs a compileErr
func err(msg ...interface{}) *compileErr {
	return &compileErr{0, fmt.Sprint(msg...)}
}

func errp(pos token.Pos, msg ...interface{}) *compileErr {
	return &compileErr{pos, fmt.Sprint(msg)}
}

// type string for value i
func typ(i interface{}) string {
	return reflect.TypeOf(reflect.ValueOf(i).Interface()).String()
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}
