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

func RunStack() {
	for _, r := range stack {
		go r.Run()
	}
	stack = nil
}
