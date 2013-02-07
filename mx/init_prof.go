package mx

// File: initialization of CPU and memory profiling.
// Author: Arne Vansteenkiste

import (
	"github.com/barnex/cuda5/cuda"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime/pprof"
)

// called by init()
func initCpuProf() {
	if *Flag_cpuprof {
		// start CPU profile to file
		fname := OD + "cpu.pprof"
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
		Log("memory profile enabled")
		AtExit(func() {
			fname := OD + "mem.pprof"
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

// Configuration for GPU profile output.
const CUDA_PROFILE_CONFIG = `
gpustarttimestamp
instructions
streamid
`

// called by init()
func initGPUProf() {
	if *Flag_gpuprof {
		PanicErr(os.Setenv("CUDA_PROFILE", "1"))
		PanicErr(os.Setenv("CUDA_PROFILE_CSV", "1"))
		out := OD + "gpuprofile.csv"
		Log("writing GPU profile to", out)
		PanicErr(os.Setenv("CUDA_PROFILE_LOG", out))
		cfgfile := OD + "cudaprof.cfg"
		PanicErr(os.Setenv("CUDA_PROFILE_CONFIG", cfgfile))
		FatalErr(ioutil.WriteFile(cfgfile, []byte(CUDA_PROFILE_CONFIG), 0666), "gpuprof")
		AtExit(cuda.DeviceReset)
	}
}

// Exec command and write output to outfile.
func saveCmdOutput(outfile string, cmd string, args ...string) {
	Log("exec:", cmd, args, ">", outfile)
	out, err := exec.Command(cmd, args...).Output() // TODO: stderr is ignored
	if err != nil {
		Logf("exec %v %v: %v: %v", cmd, args, err, string(out))
	} else {
		e := ioutil.WriteFile(outfile, out, 0666)
		LogErr(e, "writing", outfile)
	}
}

// path to the executable.
func procSelfExe() string {
	me, err := os.Readlink("/proc/self/exe")
	PanicErr(err)
	return me
}
