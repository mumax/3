/*
engine does the simulation bookkeeping, I/O and GUI.

space-dependence:
value: space-independent
param: region-dependent parameter (always input)
field: fully space-dependent field

TODO: godoc everything

*/
package engine

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
)

const VERSION = "mumax 3.11"

var UNAME = fmt.Sprintf("%s [%s_%s %s(%s) CUDA-%d.%d]",
	VERSION, runtime.GOOS, runtime.GOARCH, runtime.Version(), runtime.Compiler,
	cu.CUDA_VERSION/1000, (cu.CUDA_VERSION%1000)/10)

var StartTime = time.Now()

var (
	busyLock sync.Mutex
	busy     bool // are we so busy we can't respond from run loop? (e.g. calc kernel)
)

// We set SetBusy(true) when the simulation is too busy too accept GUI input on Inject channel.
// E.g. during kernel init.
func SetBusy(b bool) {
	busyLock.Lock()
	defer busyLock.Unlock()
	busy = b
}

func GetBusy() bool {
	busyLock.Lock()
	defer busyLock.Unlock()
	return busy
}

// Cleanly exits the simulation, assuring all output is flushed.
func Close() {
	drainOutput()
	LogUsedRefs()
	Table.flush()
	if logfile != nil {
		logfile.Close()
	}
	if bibfile != nil {
		bibfile.Close()
	}
	if *Flag_sync {
		timer.Print(os.Stdout)
	}

}
