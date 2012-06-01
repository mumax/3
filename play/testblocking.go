package main

import (
	"github.com/barnex/cuda4/cu"
	"log"
	"unsafe"
	"runtime"
)

var (ctx cu.Context)

func init() {
	log.SetFlags(log.Lmicroseconds)

	log.Println("cu.Init")
	cu.Init(0)

	log.Println("cu.CtxCreate")
	ctx = cu.CtxCreate(cu.CTX_SCHED_YIELD, 0)

	log.Println("ctx.SetCurrent()")
	ctx.SetCurrent()

	// Workaround to try to set the same context for all go threads,
	// avoids CUDA_INVALID_CONTEXT
	N := runtime.GOMAXPROCS(-1)
	done := make(chan int, N)
	for i:=0; i<N; i++{
		go func(){
			log.Println("ctx.SetCurrent()")
			ctx.SetCurrent()
			done <- 1
		}()
	}
	for i:=0; i<N; i++{
		<-done
	}
	log.Println("set", N, "contexts")
}

func main() {

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
	load = 10000

	var n int
	n = N / 2

	args := []unsafe.Pointer{unsafe.Pointer(&array), unsafe.Pointer(&load), unsafe.Pointer(&n)}
	block := 128
	grid := DivUp(N, block)
	shmem := 0
	log.Println("block:", block, "grid", grid)
	done := make(chan int)
	go func() {
		// ctx.SetCurrent()
		stream := cu.StreamCreate()
		log.Println("start")
		cu.LaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, stream, args)
		stream.Synchronize()
		log.Println("done")
		done <- 1
	}()
	
	log.Println("waiting for <-done")
	<-done
	log.Println("<-done OK")

	cu.MemcpyDtoH(aptr, A, bytes)
	//fmt.Println(a)
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
