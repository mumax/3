package core

import "testing"

func TestRWMutex(t *testing.T) {
	N := 10
	m := NewRWMutex(N)

	go func() {
		m.RLock(0, 5)
		Log("r", 0, 5)

		m.RLock(5, 10)
		Log("r", 5, 10)

		m.RLock(0, 5)
		Log("r", 0, 5)

		m.RLock(5, 10)
		Log("r", 5, 10)
	}()

	m.Lock(0, 5)
	Log("w", 0, 5)

	m.Lock(5, 10)
	Log("w", 5, 10)

	m.Lock(0, 5)
	Log("w", 0, 5)

	m.Lock(5, 10)
	Log("w", 5, 10)

}
