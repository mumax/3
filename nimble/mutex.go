package nimble

type mutex interface {
	next(delta int)
	done()
	lockedRange() (start, stop int)
	isLocked() bool
	//delta(Δstart, Δstop int)
}
