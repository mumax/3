package core

import "testing"

func TestRWMutex(t *testing.T) {
	N := 100
	m := NewRWMutex(N)

	m.c = 5
	m.d = 10

	m.Lock(0, 50)
	m.Lock(50, 100)
	m.Lock(0, 0)
	m.Lock(0, 50)
}
