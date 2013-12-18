package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"log"
	"runtime"
	"time"
)

const VERSION = "mumax3.4.0"

var UNAME = VERSION + " " + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + " (" + runtime.Compiler + ")"

var StartTime = time.Now()

// check if m is set
func checkM() {
	checkMesh()
	if M.Buffer().DevPtr(0) == nil {
		util.Fatal("need to initialize magnetization first")
	}
	if cuda.MaxVecNorm(M.Buffer()) == 0 {
		util.Fatal("need to initialize magnetization first")
	}
}

// Cleanly exits the simulation, assuring all output is flushed.
func Close() {
	drainOutput()
	Table.flush()
	if logfile != nil {
		logfile.Close()
	}

	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	log.Println("Total memory allocation", memstats.TotalAlloc/(1024), "KiB")

	// debug. TODO: rm
	//	for n, p := range params {
	//		if u, ok := p.(interface {
	//			nUpload() int
	//		}); ok {
	//			log.Println(n, "\t:\t", u.nUpload(), "uploads")
	//		}
	//	}
}

// TODO
//func sanitycheck() {
//	if Msat() == 0 {
//		log.Fatal("Msat should be nonzero")
//	}
//}
