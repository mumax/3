package nc

// Garbageman recycles garbage slices.

import (
	"log"
	"sync"
	"runtime"
	"fmt"
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
		if count == 0 {
			select {
			case fresh <- g:
				if Debug{log.Println("recycle", &g[0])}
			default:
				log.Println("spilling", &g[0])
				delete(refcount, &g[0]) // allow it to be GC'd
			}
		}
		if count < 0 {
			Panic("reference count = ", count)
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
	refcount[&s[X][0]] += count
	refcount[&s[Y][0]] += count
	refcount[&s[Z][0]] += count
	lock.Unlock()
}
func decr(s []float32) (count int) {
	count = refcount[&s[0]]
	count--
	refcount[&s[0]] = count
	return count
}

func Buffer() []float32 {
	log.Println("buffer for", caller())
	select {
	default:
		log.Println("alloc")
		return make([]float32, WarpLen())
	case f := <-fresh:
		return f
	}
	return nil // silence gc
}

func caller() string{
	_, file, line, _ := runtime.Caller(2)
	return fmt.Sprint(file, ":", line)
}

func Buffer3() [3][]float32 {
	log.Println("buffer3 for", caller())
	return [3][]float32{Buffer(), Buffer(), Buffer()}
}

func initGarbageman() {
	garbage = make(chan []float32, 0)
	fresh = make(chan []float32, 100*NumWarp()) // need big buffer to avoid spilling
	go runGC()
}

func Recycle(garbages ...[]float32) {
	for _, g := range garbages {
		log.Println("recycle for", caller())
		// debug: allow to trace the bad guy
		{
			lock.Lock()
			if refcount[&g[0]] == 0 {
				Panic("reference count=0")
			}
			lock.Unlock()
		}

		garbage <- g
	}
}

func Recycle3(garbages ...[3][]float32) {
	for _, g := range garbages {
		log.Println("recycle3 for", caller())
		Recycle(g[X], g[Y], g[Z])
	}
}
