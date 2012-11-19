package nimble

import "fmt"

type Runner interface {
	Run()
}

var stack []Runner

func Stack(r Runner) {
	for _, s := range stack {
		if s == r {
			panic(fmt.Errorf("stack: runner already stacked"))
		}
	}
	stack = append(stack, r)
}

func StackFunc(f func()){
	Stack(funcRunner(f))
}

func RunStack() {
	for _, r := range stack {
		go r.Run()
	}
	stack = nil
}

type funcRunner func()

func(f funcRunner)Run(){f()}
