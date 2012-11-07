package nimble

type Runner interface {
	Run()
}

var stack []Runner

func Stack(r Runner) {
	stack = append(stack, r)
}

func RunStack() {
	for _, r := range stack {
		go r.Run()
	}
	stack = nil
}
