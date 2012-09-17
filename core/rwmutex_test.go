package core

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestRWMutex(t *testing.T) {
	N := 10
	a := make([]int, N)
	frames := 10
	m := NewRWMutex(N)

	go func() {
		count := 0
		for i := 0; i < frames; i++ {
			prev := 0
			for j := 1; j <= N; j++ {
				m.RLock(prev, j)
				fmt.Printf("                   R % 3d % 3d: %d\n", prev, j, a[prev])
				if count != a[prev] {
					t.Error("got", a[prev], "expected", count)
				}
				count++
				prev = j
				time.Sleep(time.Duration(rand.Int63n(10000)))
			}
		}
	}()

	count := 0
	for i := 0; i < frames; i++ {
		prev := 0
		for j := 1; j <= N; j++ {
			m.Lock(prev, j)
			fmt.Printf("W % 3d % 3d: %d\n", prev, j, count)
			a[prev] = count
			count++
			prev = j
			time.Sleep(time.Duration(rand.Int63n(1000)))
		}
	}
}
