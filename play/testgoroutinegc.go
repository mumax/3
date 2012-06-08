package main

import (
	"fmt"
	"runtime"
)

func main() {
	for {
		go func() { <-make(chan int) }()
		runtime.Gosched()
		runtime.GC()
		fmt.Println(runtime.NumGoroutine())
	}
}
