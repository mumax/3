package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"runtime"
	"time"
)

const VERSION = "mumax3.4.0"

var UNAME = VERSION + " " + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + " (" + runtime.Compiler + ")"

var StartTime = time.Now()

var (
	globalmesh data.Mesh     // mesh for m and everything that has the same size
	M          magnetization // reduced magnetization (unit length)
	B_eff      setter        // total effective field
)

func init() {
	DeclFunc("SetGridSize", SetGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("SetCellSize", SetCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclFunc("SetPBC", SetPBC, `Sets number of repetitions in X,Y,Z`)
	DeclLValue("m", &M, `Reduced magnetization (unit length)`)
	B_eff.init(VECTOR, &globalmesh, "B_eff", "T", "Effective field", SetEffectiveField)
}

// Sets dst to the current effective field (T).
func SetEffectiveField(dst *data.Slice) {
	B_demag.Set(dst)  // set to B_demag...
	B_exch.AddTo(dst) // ...then add other terms
	B_anis.AddTo(dst)
	B_ext.AddTo(dst)
	B_therm.AddTo(dst)
}

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
	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	//log.Println("Total memory allocation", memstats.TotalAlloc/(1024), "KiB")

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
