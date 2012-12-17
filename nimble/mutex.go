package nimble

type mutex interface {
	lockNext(delta int) // TODO: return locked range
	unlock()
	lockedRange() (start, stop int)
	isLocked() bool
}
