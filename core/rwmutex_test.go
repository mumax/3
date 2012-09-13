package core

import "testing"

func TestRWMutex(t *testing.T) {
	N := 100
	var m RWMutex
	m.N = N

	m.Lock(0, 50)
	m.Lock(50, 100)
	m.Lock(0, 0)
	m.Lock(0, 50)
}
