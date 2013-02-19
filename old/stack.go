package nimble

import (
	"fmt"
	"sync"
)

// Runner's Run method typically starts an infinite loop
// reading from some inputs and writing processed data to some outputs.
type Runner interface {
	Run()
}

var (
	stack     []Runner
	stacklock sync.Mutex
)

// Stack schedules a Runner to run when RunStack is called.
func Stack(r Runner) {
	stacklock.Lock()
	defer stacklock.Unlock()

	for _, s := range stack {
		if s == r {
			panic(fmt.Errorf("stack: runner already stacked"))
		}
	}
	stack = append(stack, r)
}

// Stack schedules a func() to run when RunStack is called.
func StackFunc(f func()) {
	Stack(funcRunner(f))
}

// RunStack runs all Runners stacked by Stack.
func RunStack() {
	stacklock.Lock()
	defer stacklock.Unlock()
	for _, r := range stack {
		go r.Run()
	}
	stack = nil
}

// wraps a func() in a Runner
type funcRunner func()

func (f funcRunner) Run() { f() }
