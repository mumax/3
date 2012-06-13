package nc

// Garbageman recycles garbage slices.

import (
	"log"
	"sync"
)

var (
	fresh    chan []float32 // used as fifo, replace by hand-written stack!
	refcount = make(map[*float32]int)
	NumAlloc int
	lock     sync.Mutex
)

func incr(s []float32, count int) {
	lock.Lock()
	defer lock.Unlock()
	if prev, ok := refcount[&s[0]]; ok {
		refcount[&s[0]] = prev + count
	}
	Assert(len(refcount) == NumAlloc)
}

func incr3(s [3][]float32, count int) {
	lock.Lock()
	defer lock.Unlock()
	for c := 0; c < 3; c++ {
		if prev, ok := refcount[&s[c][0]]; ok {
			refcount[&s[c][0]] = prev + count
		}
	}
}

func Buffer() []float32 {
	lock.Lock()
	defer lock.Unlock()

	select {
	case f := <-fresh:
		log.Println("re-use", &f[0])
		return f
	default:
		slice := make([]float32, WarpLen())
		NumAlloc++
		log.Println("alloc", &slice[0])
		refcount[&slice[0]] = 0
		return slice
	}
	return nil // silence gc
}

//func caller() string {
//	_, file, line, _ := runtime.Caller(2)
//	return fmt.Sprint(file, ":", line)
//}

func Buffer3() [3][]float32 {
	return [3][]float32{Buffer(), Buffer(), Buffer()}
}

func initGarbageman() {
	fresh = make(chan []float32, 5*NumWarp()) // need big buffer to avoid spilling
}

func Recycle(garbages ...[]float32) {
	lock.Lock()
	defer lock.Unlock()

	for _, g := range garbages {

		count, ok := refcount[&g[0]]
		if !ok {
			continue // slice does not originate from here
		}
		if count == 0 { // can be recycled
			select {
			case fresh <- g:
				log.Println("recycle", &g[0])
			default:
				log.Println("spilling", &g[0])
				delete(refcount, &g[0]) // allow it to be GC'd TODO: spilltest
			}
		} else { // cannot be recycled, just yet
			refcount[&g[0]] = count - 1
		}
	}
}

func Recycle3(garbages ...[3][]float32) {
	for _, g := range garbages {
		Recycle(g[X], g[Y], g[Z])
	}
}
