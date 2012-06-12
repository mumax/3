package nc

// Garbageman recycles garbage slices.

import (
	"fmt"
	"log"
	"runtime"
	"sync"
)

var (
	garbage  chan []float32
	fresh    chan []float32
	refcount = make(map[*float32]int)
	lock     sync.Mutex
)

func runGC() {
	log.Println("GC started")
	for {
		g := <-garbage
		count := decr(g)
		Assert(count > -2)
		if count == -1 {
			incr(g, 1) // set to 0 for next use.
			select {
			case fresh <- g:
				log.Println("recycle", &g[0])
			default:
				log.Println("spilling", &g[0])
				delete(refcount, &g[0]) // allow it to be GC'd
			}
		}
	}
}

func incr(s []float32, count int) {
	lock.Lock()
	refcount[&s[0]] += count
	lock.Unlock()
}

func incr3(s [3][]float32, count int) {
	lock.Lock()
	defer lock.Unlock()
	refcount[&s[X][0]] += count
	refcount[&s[Y][0]] += count
	refcount[&s[Z][0]] += count
}

func decr(s []float32) (count int) {
	lock.Lock()
	defer lock.Unlock()
	count = refcount[&s[0]]
	count--
	refcount[&s[0]] = count
	return count
}

func count(s []float32) (count int) {
	lock.Lock()
	defer lock.Unlock()
	return refcount[&s[0]]
}

func Buffer() []float32 {
	lock.Lock()
	defer lock.Unlock()
	//log.Println("buffer for", caller())
	//log.Println(refcount)
	select {
	default:
		slice := make([]float32, WarpLen())
		log.Println("alloc", &slice[0])
		return slice
	case f := <-fresh:
		log.Println("re-use", &f[0])
		return f
	}
	return nil // silence gc
}

func caller() string {
	_, file, line, _ := runtime.Caller(2)
	return fmt.Sprint(file, ":", line)
}

func Buffer3() [3][]float32 {
	//log.Println("buffer3 for", caller())
	return [3][]float32{Buffer(), Buffer(), Buffer()}
}

func initGarbageman() {
	garbage = make(chan []float32, 0)
	fresh = make(chan []float32, 5*NumWarp()) // need big buffer to avoid spilling
	go runGC()
}

func Recycle(garbages ...[]float32) {
	for _, g := range garbages {

		if count(g) < 0 {
			Panic("ref count", count(g))
		}

		garbage <- g
	}
}

func Recycle3(garbages ...[3][]float32) {
	for _, g := range garbages {
		//log.Println("recycle3 for", caller())
		Recycle(g[X], g[Y], g[Z])
	}
}
