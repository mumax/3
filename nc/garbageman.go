package nc

// Garbageman recycles garbage slices.

import (
	"sync"
)

var (
	fresh    chan []float32 // used as fifo, replace by hand-written stack!
	refcount = make(map[*float32]int)
	NumAlloc int
	lock     sync.Mutex
)

// increment the reference count by count.
func incr(s []float32, count int) {
	lock.Lock()
	defer lock.Unlock()
	if prev, ok := refcount[&s[0]]; ok {
		refcount[&s[0]] = prev + count
	}
	Assert(len(refcount) == NumAlloc)
}

// increment the reference count by count.
func incr3(s [3][]float32, count int) {
	lock.Lock()
	defer lock.Unlock()
	for c := 0; c < 3; c++ {
		if prev, ok := refcount[&s[c][0]]; ok {
			refcount[&s[c][0]] = prev + count
		}
	}
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
	select {
	case f := <-fresh:
		//log.Println("re-use", &f[0])
		return f
	default:
		slice := make([]float32, WarpLen())
		NumAlloc++
		//log.Println("alloc", &slice[0])
		refcount[&slice[0]] = 0
		return slice
	}
	return nil // silence gc
}

func initGarbageman() {
	fresh = make(chan []float32, 10*NumWarp()) // need big buffer to avoid spilling
}

func Recycle(garbages ...[]float32) {
	lock.Lock()
	defer lock.Unlock()

	for _, g := range garbages {
		//log.Println("Recycle(", &g[0], ")")
		count, ok := refcount[&g[0]]
		if !ok {
			//log.Println("skipping", &g[0])
			continue // slice does not originate from here
		}
		if count == 0 { // can be recycled
			select {
			case fresh <- g:
				//log.Println("recycling", &g[0])
			default:
				//log.Println("spilling", &g[0])
				delete(refcount, &g[0]) // allow it to be GC'd TODO: spilltest
			}
		} else { // cannot be recycled, just yet
			//log.Println("decrementing", &g[0], ":", count-1)
			refcount[&g[0]] = count - 1
		}
	}
}

func Recycle3(garbages ...[3][]float32) {
	for _, g := range garbages {
		Recycle(g[X], g[Y], g[Z])
	}
}
