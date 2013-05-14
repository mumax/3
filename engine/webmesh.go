package engine

// Handlers for mesh setting through web interface

import (
	"code.google.com/p/mx3/data"
	"log"
	"net/http"
	"strconv"
)

// TODO replace by rpc call
func setmesh(w http.ResponseWriter, r *http.Request) {

	inject <- pauseFn

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

	injectAndWait(func() {
		var mh *data.Slice
		if globalmesh.Size() != [3]int{} { // if set. TODO: nicer api
			mh = M.buffer.HostCopy()
		}
		SetMesh(N[0], N[1], N[2], c[0], c[1], c[2])
		if mh != nil {
			M.Set(mh)
		}
	})

	http.Redirect(w, r, "/", http.StatusFound)
}

// for all keys, fetch and parse integer values from the http form.
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

// for all keys, fetch and parse float values from the http form.
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
