package core

import "testing"

func TestRWMutex(t *testing.T) {
	N := 100
	m := NewRWMutex(N)


	m.Lock(0, 1)

	go func() {
		m.RLock(0, 1)
		m.RLock(1, 2)
		m.RLock(2, 3)
		m.RLock(4, 5)
	}()

	m.Lock(1, 2)
	m.Lock(2, 3)
	m.Lock(3, 4)
}
