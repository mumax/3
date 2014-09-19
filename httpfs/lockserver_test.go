package httpfs

import (
	"fmt"
	"testing"
	"time"
)

func TestLockServer(t *testing.T) {
	D := 50 * time.Millisecond
	l := NewLockServer(D)

	// user can indefinitely acquire lock
	for i := 0; i < 100; i++ {
		if !l.Lock("Arne", "milk") {
			t.Fail()
		}
	}

	// other user can't acquire it now
	if l.Lock("Bartel", "milk") {
		t.Fail()
	}

	// lease time expired, other user can acquire
	time.Sleep(D)
	if !l.Lock("Bartel", "milk") {
		t.Fail()
	}
	if l.Lock("Arne", "milk") {
		t.Fail()
	}

	// but can acquire other lock
	if !l.Lock("Arne", "chocoloate") {
		t.Fail()
	}

	time.Sleep(D)
	if !l.Lock("Bartel", "chocolate") {
		t.Fail()
	}

	// stress test, many users
	for i := 0; i < 100000; i++ {
		if l.Lock(fmt.Sprint(i), "chocolate") {
			t.Fail()
		}
	}

	// stress test, many keys
	for i := 0; i < 100000; i++ {
		if !l.Lock("Arne", fmt.Sprint(i)) {
			t.Fail()
		}
	}
}
