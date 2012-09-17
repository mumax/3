package core

type RMutex struct {
	rw   *RWMutex
	c, d int
}

// Lock for reading [start, stop[.
// Automatically unlocks the previous interval.
// RLock(0, 0) can be used to explicitly unlock.
func (m *RMutex) RLock(start, stop int) {

}
