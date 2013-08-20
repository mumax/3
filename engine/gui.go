package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/gui"
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"time"
)

var (
	GUI    *gui.Doc
	quants = make(map[string]Getter) // INTERNAL maps quantity names to downloadable data. E.g. for rendering
)

func init() {
	quants["m"] = &M
	quants["mFFT"] = &fftmPower{} // for the web interface we display FFT amplitude
	quants["B_anis"] = &B_anis
	quants["Ku1"] = &Ku1
	quants["Kc1"] = &Kc1
	quants["anisU"] = &AnisU
	quants["anisC1"] = &AnisC1
	quants["anisC2"] = &AnisC2
	quants["regions"] = &regions
	quants["B_demag"] = &B_demag
	quants["B_eff"] = &B_eff
	quants["torque"] = &Torque
	quants["B_exch"] = &B_exch
	quants["lltorque"] = &LLTorque
	quants["jpol"] = &JPol
	quants["sttorque"] = &STTorque
}

// Start web gui on given port, does not block.
func GoServe(port string) {

	data := &struct{ Quants *map[string]Getter }{&quants}
	GUI = gui.NewDoc(templText, data)

	http.Handle("/", GUI)
	http.HandleFunc("/render/", serveRender)

	GUI.SetValue("gpu", fmt.Sprint(cuda.DevName, " (", (cuda.TotalMem)/(1024*1024), "MB)", ", CUDA ", cuda.Version))
	hostname, _ := os.Hostname()
	GUI.SetValue("hostname", hostname)
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
