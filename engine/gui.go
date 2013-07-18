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

	//http.HandleFunc("/render/", render)

	GUI.SetValue("gpu", fmt.Sprint(cuda.DevName, " (", (cuda.TotalMem)/(1024*1024), "MB)", ", CUDA ", cuda.Version))

	GUI.OnClick("break", Pause)
	GUI.OnClick("run", func() { Inject <- func() { Run(GUI.Value("runtime").(float64)) } })
	GUI.OnClick("steps", func() { Inject <- func() { Steps(GUI.Value("runsteps").(int)) } })
	GUI.OnChange("fixdt", func() { Inject <- func() { Solver.FixDt = GUI.Value("fixdt").(bool) } })
	GUI.OnChange("mindt", func() { Inject <- func() { Solver.MinDt = GUI.Value("mindt").(float64) } })
	GUI.OnChange("maxdt", func() { Inject <- func() { Solver.MaxDt = GUI.Value("maxdt").(float64) } })

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
	GUI.SetValue("time", float32(Time))
	GUI.SetValue("dt", float32(Solver.Dt_si))
	GUI.SetValue("step", float32(Solver.NSteps))
	GUI.SetValue("lasterr", float32(Solver.LastErr))
}
