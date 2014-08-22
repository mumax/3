// package prof provides run-time profiling.
package prof

import (
	"github.com/mumax/3/util"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"runtime/pprof"
)

func InitCPU(OD string) {
	// start CPU profile to file
	fname := OD + "cpu.pprof"
	f, err := os.Create(fname)
	util.FatalErr(err)
	err = pprof.StartCPUProfile(f)
	util.FatalErr(err)
	log.Println("writing CPU profile to", fname)

	// at exit: exec go tool pprof to generate SVG output
	AtExit(func() {
		pprof.StopCPUProfile()
		me := procSelfExe()
		outfile := fname + ".svg"
		saveCmdOutput(outfile, "go", "tool", "pprof", "-svg", me, fname)
	})
}

func InitMem(OD string) {
	log.Println("memory profile enabled")
	AtExit(func() {
		fname := OD + "mem.pprof"
		f, err := os.Create(fname)
		defer f.Close()
		util.LogErr(err, "memory profile") // during cleanup, should not panic/exit
		log.Println("writing memory profile to", fname)
		util.LogErr(pprof.WriteHeapProfile(f), "memory profile")
		me := procSelfExe()
		outfile := fname + ".svg"
		saveCmdOutput(outfile, "go", "tool", "pprof", "-svg", "--inuse_objects", me, fname)
	})
}

// Exec command and write output to outfile.
func saveCmdOutput(outfile string, cmd string, args ...string) {
	log.Println("exec:", cmd, args, ">", outfile)
	out, err := exec.Command(cmd, args...).Output() // TODO: stderr is ignored
	if err != nil {
		log.Printf("exec %v %v: %v: %v", cmd, args, err, string(out))
	}
	// on error: write anyway, clobbers output file.
	e := ioutil.WriteFile(outfile, out, 0666)
	util.LogErr(e, "writing", outfile)
}

// path to the executable.
func procSelfExe() string {
	me, err := os.Readlink("/proc/self/exe")
	util.PanicErr(err)
	return me
}
