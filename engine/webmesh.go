package engine

// Handlers for mesh setting through web interface

import (
	"log"
	"net/http"
	"strconv"
)

func setmesh(w http.ResponseWriter, r *http.Request) {
	pause()
	ui.Lock()
	defer ui.Unlock()

	log.Println("setmesh")

	N, err := formInts(r, "gridsizex", "gridsizey", "gridsizez")
	if err != nil {
		http.Error(w, "grid size: "+err.Error(), 400)
		return
	}
	c, err2 := formFloats(r, "cellsizex", "cellsizey", "cellsizez")
	if err2 != nil {
		http.Error(w, "cell size: "+err2.Error(), 400)
		return
	}

	SetMesh(N[0], N[1], N[2], c[0], c[1], c[2])

	http.Redirect(w, r, "/", http.StatusFound)

}

func formInts(r *http.Request, key ...string) ([]int, error) {
	vals := make([]int, len(key))
	for i, k := range key {
		v, err := strconv.Atoi(r.FormValue(k))
		if err != nil {
			return nil, err
		}
		vals[i] = v
	}
	return vals, nil
}

func formFloats(r *http.Request, key ...string) ([]float64, error) {
	vals := make([]float64, len(key))
	for i, k := range key {
		v, err := strconv.ParseFloat(r.FormValue(k), 64)
		if err != nil {
			return nil, err
		}
		vals[i] = v
	}
	return vals, nil
}
