package core

import (
	"math/rand"
	"testing"
	"time"
)

func TestRWMutex(t *testing.T) {
	N := 10
	a := make([]int, N)
	frames := 3
	m := NewRWMutex(N)

	go func() {
		count := 0
		for i := 0; i < frames; i++ {
			prev := 0
			for j := 1; j <= N; j++ {
				m.RLock(prev, j)
				if count != a[prev] {
					t.Error("got", a[prev], "expected", count)
				}
				count++
				prev = j
				time.Sleep(time.Duration(rand.Int63n(1000)))
			}
		}
	}()

	count := 0
	for i := 0; i < frames; i++ {
		prev := 0
		for j := 1; j <= N; j++ {
			m.Lock(prev, j)
			a[prev] = count
			count++
			prev = j
			time.Sleep(time.Duration(rand.Int63n(1000)))
		}
	}
}
