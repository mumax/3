package core

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

const D = 100 * time.Microsecond
const P = 0.95

// Write consecutive numbers in the array
// and test if read receives them in the correct order.
func TestRWMutex(t *testing.T) {
	N := 3
	a := make([]int, N)
	frames := 1000
	m := NewRWMutex(N)
	r1 := m.NewReader()
	r2 := m.NewReader()

	go write(m, a, N, frames)
	time.Sleep(time.Millisecond)
	go read(r1, a, N, frames, t)
	read(r2, a, N, frames, t)
}

func TestRWMutex_try(t *testing.T) {
	N := 3
	a := make([]int, N)
	frames := 1000
	m := NewRWMutex(N)
	r1 := m.NewReader()
	r2 := m.NewReader()

	go write(m, a, N, frames)
	time.Sleep(time.Millisecond)
	go read_try(r1, a, N, frames, t)
	read_try(r2, a, N, frames, t)
}

func read(m *RMutex, a []int, N, frames int, t *testing.T) {
	count := 0
	for i := 0; i < frames; i++ {
		prev := 0
		for j := 1; j <= N; j += 1 {
			m.RLock(prev, j)
			fmt.Printf("                   R % 3d % 3d: %d\n", prev, j, a[prev])
			if count != a[prev] {
				t.Error("got", a[prev], "expected", count)
			}
			count++
			prev = j
			if rand.Float32() > P {
				time.Sleep(D)
			}
		}
		if rand.Float32() > P {
			time.Sleep(D)
		}
	}
}

func read_try(m *RMutex, a []int, N, frames int, t *testing.T) {
	count := 0
	for i := 0; i < frames; i++ {
		prev := 0
		for j := 1; j <= N; j += 1 {
			for !m.TryRLock(prev, j) {
				time.Sleep(1 * time.Microsecond)
			}
			fmt.Printf("                   R % 3d % 3d: %d\n", prev, j, a[prev])
			if count != a[prev] {
				t.Error("got", a[prev], "expected", count)
			}
			count++
			prev = j
			if rand.Float32() > P {
				time.Sleep(D)
			}
		}
		if rand.Float32() > P {
			time.Sleep(D)
		}
	}
}

func write(m *RWMutex, a []int, N, frames int) {
	count := 0
	for i := 0; i < frames; i++ {
		prev := 0
		for j := 1; j <= N; j++ {
			m.WLock(prev, j)
			fmt.Printf("W % 3d % 3d: %d\n", prev, j, count)
			a[prev] = count
			count++
			prev = j
			if rand.Float32() > P {
				time.Sleep(D)
			}
		}
		if rand.Float32() > P {
			time.Sleep(D)
		}
	}
	m.WLock(0, 0) // unlocks
}
