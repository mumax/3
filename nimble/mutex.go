package nimble

type mutex interface {
	Delta(Δstart, Δstop int)
	Next(delta int)
	Done()
	Range() (start, stop int)
}
