package main

import (
	"github.com/barnex/cuda4/cu"
	"log"
	"runtime"
	"time"
	"unsafe"
)

var (
	ctx cu.Context
)

func init() {
	log.SetFlags(log.Lmicroseconds)

	log.Println("cu.Init")
	cu.Init(0)

	log.Println("cu.CtxCreate")
	ctx = cu.CtxCreate(cu.CTX_SCHED_YIELD, 0)
}

func main() {

	ctx.SetCurrent()

	log.Println(`cu.ModuleLoad("testmodule.ptx")`)
	mod := cu.ModuleLoad("testmodule.ptx")
	log.Println(`mod.GetFunction("generateLoad")`)
	f := mod.GetFunction("generateLoad")

	N := 1000000
	bytes := cu.SIZEOF_FLOAT32 * int64(N)
	a := make([]float32, N)
	A := cu.MemAlloc(bytes)
	defer A.Free()
	aptr := unsafe.Pointer(&a[0])
	cu.MemcpyHtoD(A, aptr, bytes)

	var array uintptr
	array = uintptr(A)

	var load int
	load = 1000

	var n int
	n = N / 2

	args := []unsafe.Pointer{unsafe.Pointer(&array), unsafe.Pointer(&load), unsafe.Pointer(&n)}
	block := 128
	grid := DivUp(N, block)
	shmem := 0
	log.Println("block:", block, "grid", grid)
	done := make(chan int)
	Nthr := 16

	for I := 0; I < Nthr; I++ {
		go func(i int) {
			ctx.SetCurrent()
			stream := cu.StreamCreate()
			log.Println("start", i)
			cu.LaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, stream, args)
			log.Println("waitforsync", i)
			runtime.Gosched()
			stream.Synchronize()
			log.Println("done", i)
			done <- 1
		}(I)
	}
	go func() {
		for {
			log.Println("tick")
			time.Sleep(100 * 1e6)
		}
	}()
	log.Println("waiting for ", Nthr, " x <-done")

	for I := 0; I < Nthr; I++ {
		<-done
	}
	log.Println("<-done OK")

	cu.MemcpyDtoH(aptr, A, bytes)
	//fmt.Println(a)
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
