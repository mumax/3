package mx

// File: initialization of CPU and memory profiling.
// Author: Arne Vansteenkiste

import (
	"io/ioutil"
	"os"
	"os/exec"
	"runtime/pprof"
)

// called by init()
func initCpuProf() {
	if *Flag_cpuprof {
		// start CPU profile to file
		fname := OD + "/cpu.pprof"
		f, err := os.Create(fname)
		FatalErr(err, "CPU profile")
		err = pprof.StartCPUProfile(f)
		FatalErr(err, "CPU profile")
		Log("writing CPU profile to", fname)

		// at exit: exec go tool pprof to generate SVG output
		AtExit(func() {
			pprof.StopCPUProfile()
			me := procSelfExe()
			outfile := fname + ".svg"
			saveCmdOutput(outfile, "go", "tool", "pprof", "-svg", me, fname)
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
			LogErr(err, "memory profile") // during cleanup, should not panic/exit
			Log("writing memory profile to", fname)
			LogErr(pprof.WriteHeapProfile(f), "memory profile")
			me := procSelfExe()
			outfile := fname + ".svg"
			saveCmdOutput(outfile, "go", "tool", "pprof", "-svg", "--inuse_objects", me, fname)
		})
	}
}

// TODO: cuda profiling: set env variables, perhaps run cli profiler.

// Exec command and write output to outfile.
func saveCmdOutput(outfile string, cmd string, args ...string) {
	Log("exec:", cmd, args, ">", outfile)
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		Logf("exec %v %v: %v: %v", cmd, args, err, string(out))
	} else {
		Logf("writing %v: %v", outfile, ioutil.WriteFile(outfile, out, 0666))
	}
}

// path to the executable.
func procSelfExe() string {
	me, err := os.Readlink("/proc/self/exe")
	PanicErr(err)
	return me
}
