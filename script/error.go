package script

import (
	"fmt"
	"reflect"
)

// compileErr, and only compileErr will be caught by Compile and returned as an error.
type compileErr string

// implements error
func (c *compileErr) Error() string {
	return string(*c)
}

// constructs a compileErr
func err(msg ...interface{}) *compileErr {
	e := compileErr(fmt.Sprint(msg...))
	return &e
}

// type string for value i
func typ(i interface{}) string {
	return reflect.TypeOf(reflect.ValueOf(i).Interface()).String()
}
