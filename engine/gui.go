package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/gui"
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"net/http"
	"runtime"
)

var GUI = gui.NewDoc("/", templText)

// Start web gui on given port, does not block.
func GoServe(port string) {

	//http.HandleFunc("/render/", render)

	GUI.SetValue("gpu", fmt.Sprint(cuda.DevName, " (", (cuda.TotalMem)/(1024*1024), "MB)", ", CUDA ", cuda.Version))

	GUI.OnClick("break", Pause)

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	go func() {
		util.LogErr(http.ListenAndServe(port, nil))
	}()
	runtime.Gosched()
}
