package core

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// Write consecutive numbers in the array
// and test if read receives them in the correct order.
func TestRWMutex(t *testing.T) {

	N := 3
	a := make([]int, N)
	frames := 1000
	m := NewRWMutex(N)
	const D = 100 * time.Microsecond
	const P = 0.95

	// go write
	go func() {
		count := 0
		for i := 0; i < frames; i++ {
			prev := 0
			for j := 1; j <= N; j++ {
				m.Lock(prev, j)
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
		m.Lock(0, 0) // unlocks
	}()

	// read
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
