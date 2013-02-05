package mx

// File: initialization of CPU and memory profiling.
// Author: Arne Vansteenkiste

import (
	"os"
	"runtime/pprof"
)

// called by init()
func initCpuProf() {
	if *Flag_cpuprof {
		// start CPU profile to file
		fname := OD + "/cpu.pprof"
		f, err := os.Create(fname)
		FatalErr(err, "start CPU profile")
		err = pprof.StartCPUProfile(f)
		FatalErr(err, "start CPU profile")
		Log("writing CPU profile to", fname)

		// at exit: exec go tool pprof to generate SVG output
		AtExit(func() {
			pprof.StopCPUProfile()
			me := ProcSelfExe()
			outfile := fname + ".svg"
			SaveCmdOutput(outfile, "go", "tool", "pprof", "-svg", me, fname)
		})
	}
}

// called by init()
func initMemProf() {
	if *Flag_memprof {
		AtExit(func() {
			fname := OD + "/mem.pprof"
			f, err := os.Create(fname)
			defer f.Close()
			FatalErr(err, "start memory profile")
			Log("writing memory profile to", fname)
			FatalErr(pprof.WriteHeapProfile(f), "start memory profile")
			me := ProcSelfExe()
			outfile := fname + ".svg"
			SaveCmdOutput(outfile, "go", "tool", "pprof", "-svg", "--inuse_objects", me, fname)
		})
	}
}
