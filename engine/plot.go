package engine

import (
	"fmt"
	"net/http"
	"os/exec"
	"strings"
)

func servePlot(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Path[len("/plot/"):]

	u := strings.Split(url, "/") // u a:b
	if len(u) != 2 {
		http.Error(w, "need a/b", http.StatusNotFound)
	}
	a, b := u[0], u[1]

	cmd := "gnuplot"
	args := []string{"-e", fmt.Sprintf(`set format x "%%g"; set format y "%%g"; set term svg; plot "%v/datatable.txt" u %v:%v w li; set output;exit;`, OD, a, b)}
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		http.Error(w, err.Error()+"\n"+string(out), http.StatusInternalServerError)
		return
	} else {
		w.Write(out)
	}
}
