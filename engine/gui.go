package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/gui"
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"time"
)

var GUI = gui.NewDoc("/", templText)

// Start web gui on given port, does not block.
func GoServe(port string) {

	http.HandleFunc("/render/", serveRender)

	GUI.SetValue("gpu", fmt.Sprint(cuda.DevName, " (", (cuda.TotalMem)/(1024*1024), "MB)", ", CUDA ", cuda.Version))
	GUI.OnClick("break", Pause)
	GUI.OnClick("run", inj(func() { Run(GUI.Value("runtime").(float64)) }))
	GUI.OnClick("steps", inj(func() { Steps(GUI.Value("runsteps").(int)) }))
	GUI.OnChange("fixdt", inj(func() { Solver.FixDt = GUI.Value("fixdt").(float64) }))
	GUI.OnChange("mindt", inj(func() { Solver.MinDt = GUI.Value("mindt").(float64) }))
	GUI.OnChange("maxdt", inj(func() { Solver.MaxDt = GUI.Value("maxdt").(float64) }))
	GUI.OnChange("maxerr", inj(func() { Solver.MaxErr = GUI.Value("maxerr").(float64) }))

	// periodically update time, steps, etc
	go func() {
		for {
			Inject <- updateDash
			time.Sleep(100 * time.Millisecond)
		}
	}()

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	go func() {
		util.LogErr(http.ListenAndServe(port, nil))
	}()
	runtime.Gosched()
}

func updateDash() {
	GUI.SetValue("time", fmt.Sprintf("%6e", Time))
	GUI.SetValue("dt", fmt.Sprintf("%4e", Solver.Dt_si))
	GUI.SetValue("step", Solver.NSteps)
	GUI.SetValue("lasterr", fmt.Sprintf("%3e", Solver.LastErr))
	GUI.SetValue("render", "/render/m")
}

func inj(f func()) func() {
	return func() { Inject <- f }
}
