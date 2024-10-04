package engine

import (
	"bytes"
	"errors"
	"fmt"
	"github.com/mumax/3/v3/httpfs"
	"image"
	"image/png"
	"io/ioutil"
	"net/http"
	"os/exec"
	"sync/atomic"
)

var nPlots int32       // counts number of active gnuplot processes
const MAX_GNUPLOTS = 5 // maximum allowed number of gnuplot processes

func (g *guistate) servePlot(w http.ResponseWriter, r *http.Request) {

	out := []byte{}

	// handle error and return wheter err != nil.
	handle := func(err error) bool {
		if err != nil {
			w.Write(emptyIMG())
			g.Set("plotErr", err.Error()+string(out))
			return true
		} else {
			return false
		}
	}

	// limit max processes
	atomic.AddInt32(&nPlots, 1)
	defer atomic.AddInt32(&nPlots, -1)
	if atomic.LoadInt32(&nPlots) > MAX_GNUPLOTS {
		handle(errors.New("too many gnuplot processes"))
		return
	}

	a := g.StringValue("usingx")
	b := g.StringValue("usingy")

	cmd := "gnuplot"
	args := []string{"-e", fmt.Sprintf(`set format x "%%g"; set key off; set format y "%%g"; set term svg size 480,320 font 'Arial,10'; plot "-" u %v:%v w li; set output;exit;`, a, b)}
	excmd := exec.Command(cmd, args...)

	stdin, err := excmd.StdinPipe()
	if handle(err) {
		return
	}

	stdout, err := excmd.StdoutPipe()
	if handle(err) {
		return
	}

	data, err := httpfs.Read(fmt.Sprintf(`%vtable.txt`, OD()))
	if handle(err) {
		return
	}

	err = excmd.Start()
	if handle(err) {
		return
	}
	defer excmd.Wait()

	_, err = stdin.Write(data)
	if handle(err) {
		return
	}
	err = stdin.Close()
	if handle(err) {
		return
	}

	out, err = ioutil.ReadAll(stdout)
	if handle(err) {
		return
	}

	w.Header().Set("Content-Type", "image/svg+xml")
	w.Write(out)
	g.Set("plotErr", "")
	return

}

var empty_img []byte

// empty image to show if there's no plot...
func emptyIMG() []byte {
	if empty_img == nil {
		o := bytes.NewBuffer(nil)
		png.Encode(o, image.NewNRGBA(image.Rect(0, 0, 4, 4)))
		empty_img = o.Bytes()
	}
	return empty_img
}
