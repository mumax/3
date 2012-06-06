package nc

import (
	"fmt"
)

// Panics on the message.
func Panic(msg ...interface{}) {
	panic(fmt.Sprint(msg...))
}

// Checks for an IO error.
func CheckIO(err error) {
	if err != nil {
		panic(err)
	}
}
