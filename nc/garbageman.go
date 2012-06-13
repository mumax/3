package nc

// Garbageman recycles garbage slices.

import (
	"sync"
)

var (
	recycled stack
	refcount = make(map[*float32]int)
	NumAlloc int
	lock     sync.Mutex
)

// increment the reference count by count.
func incr(s []float32, count int) {
	lock.Lock()
	if prev, ok := refcount[&s[0]]; ok {
		refcount[&s[0]] = prev + count
	}
	lock.Unlock()
	//Assert(len(refcount) == NumAlloc)
}

// increment the reference count by count.
func incr3(s [3][]float32, count int) {
	lock.Lock()
	for c := 0; c < 3; c++ {
		if prev, ok := refcount[&s[c][0]]; ok {
			refcount[&s[c][0]] = prev + count
		}
	}
	lock.Unlock()
}

// Return a buffer, recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func Buffer() []float32 {
	lock.Lock()
	b := buffer()
	lock.Unlock()
	return b
}

// See Buffer()
func Buffer3() [3][]float32 {
	lock.Lock()
	b := [3][]float32{buffer(), buffer(), buffer()}
	lock.Unlock()
	return b
}

// not synchronized.
func buffer() []float32 {
	if f := recycled.pop(); f != nil {
		//log.Println("re-use", &f[0])
		return f
	}
	slice := make([]float32, WarpLen())
	NumAlloc++
	//log.Println("alloc", &slice[0])
	refcount[&slice[0]] = 0
	return slice
}

func Recycle(garbages ...[]float32) {
	lock.Lock()

	for _, g := range garbages {
		count, ok := refcount[&g[0]]
		if !ok {
			//log.Println("skipping", &g[0])
			continue // slice does not originate from here
		}
		if count == 0 { // can be recycled
			recycled.push(g)
			//log.Println("spilling", &g[0])
			//delete(refcount, &g[0]) // allow it to be GC'd TODO: spilltest
		} else { // cannot be recycled, just yet
			//log.Println("decrementing", &g[0], ":", count-1)
			refcount[&g[0]] = count - 1
		}

	}
	lock.Unlock()
}

func Recycle3(garbages ...[3][]float32) {
	for _, g := range garbages {
		Recycle(g[X], g[Y], g[Z])
	}
}

type stack [][]float32

func (s *stack) push(slice []float32) {
	(*s) = append((*s), slice)
}

func (s *stack) pop() (slice []float32) {
	if len(*s) == 0 {
		return nil
	}
	slice = (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return
}
