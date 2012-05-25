package main

import (
	"fmt"
	"runtime"
)

func main() {
	PrintHello()
}

func PrintHello() {
	fmt.Println("Nimble Cube", runtime.Compiler, runtime.GOOS, runtime.GOARCH)
}
