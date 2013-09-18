package engine

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"os/exec"
	"strings"
)

func servePlot(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Path[len("/plot/"):]

	var a, b interface{}
	u := strings.Split(url, "/") // u a:b
	if len(u) == 2 {
		a, b = u[0], u[1]
	} else {
		a, b = usingX, usingY
	}

	cmd := "gnuplot"
	args := []string{"-e", fmt.Sprintf(`set format x "%%g"; set format y "%%g"; set term png; plot "%v/datatable.txt" u %v:%v w li; set output;exit;`, OD, a, b)}
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		w.Write(emptyIMG())
		if gui_ != nil {
			gui_.SetValue("plotErr", string(out))
		}
		return
	} else {
		w.Write(out)
		gui_.SetValue("plotErr", "")
	}
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
