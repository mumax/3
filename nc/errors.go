package nc

// Error handling utilities.

import (
	"fmt"
	"log"
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

// Logs the error of non-nil.
func CheckLog(err error) {
	if err != nil {
		log.Println("[error  ]", err)
	}
}
