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
	quants = map[string]Getter{
		"m":        &M,
		"mFFT":     &fftmPower{},
		"B_anis":   &B_anis,
		"Ku1":      &Ku1,
		"Kc1":      &Kc1,
		"anisU":    &AnisU,
		"anisC1":   &AnisC1,
		"anisC2":   &AnisC2,
		"Msat":     &Msat,
		"Aex":      &Aex,
		"alpha":    &Alpha,
		"regions":  &regions,
		"B_demag":  &B_demag,
		"B_eff":    &B_eff,
		"torque":   &Torque,
		"B_exch":   &B_exch,
		"lltorque": &LLTorque,
		"jpol":     &JPol,
		"sttorque": &STTorque}
)

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
	GUI.OnChange("sel_render", func() { renderQ = GUI.Value("sel_render").(string); updateDash() })

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

var start = time.Now()

func updateDash() {
	GUI.SetValue("time", fmt.Sprintf("%6e", Time))
	GUI.SetValue("dt", fmt.Sprintf("%4e", Solver.Dt_si))
	GUI.SetValue("step", Solver.NSteps)
	GUI.SetValue("lasterr", fmt.Sprintf("%3e", Solver.LastErr))
	cachebreaker := "?" + fmt.Sprint(time.Now().Nanosecond())
	GUI.SetValue("render", "/render/"+renderQ+cachebreaker)
	GUI.SetValue("walltime", fmt.Sprint(roundt(time.Since(start))))
}

// round duration to 1s accuracy
func roundt(t time.Duration) time.Duration {
	return t - t%1e9
}

func inj(f func()) func() {
	return func() { Inject <- f }
}
