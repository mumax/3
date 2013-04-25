package engine

import (
	"net/http"
	"os/exec"
)

// handler that executes the command and returns the output
func command(cmd string, args ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		out, err := exec.Command(cmd, args...).CombinedOutput()
		w.Write(out)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}
