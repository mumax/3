package script

import (
	"fmt"
	"go/token"
	"reflect"
	"strings"
)

var Debug = false // print debug info?

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
func err(pos token.Pos, msg ...interface{}) *compileErr {
	str := fmt.Sprintln(msg...) // use Sprinln to insert spaces
	str = str[:len(str)-1]      // strip final \n
	return &compileErr{pos, str}
}

// type string for value i
func typ(i interface{}) string {
	typ := reflect.TypeOf(reflect.ValueOf(i).Interface()).String()
	if strings.HasPrefix(typ, "*ast.") {
		typ = typ[len("*ast."):]
	}
	return typ
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}

// decodes a token position in source to a line number
// and returns the line number + line code.
func pos2line(pos token.Pos, src string) string {
	if pos == 0 {
		return ""
	}
	lines := strings.Split(src, "\n")
	line := 0
	for i, b := range src {
		if token.Pos(i) == pos {
			return fmt.Sprint("line ", line, ": ", strings.Trim(lines[line], " \t")) // func{ prefix makes lines count from 1
		}
		if b == '\n' {
			line++
		}
	}
	return fmt.Sprint("position", pos) // we should not reach this
}
