package httpfs

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
)

func Serve(root, addr string) {
	log.Println("serving", root, "at", addr)
	err := http.ListenAndServe(addr, &fileHandler{path: root}) // don't use DefaultServeMux which redirects some requests behind our back.
	if err != nil {
		panic(err)
	}
}

type fileHandler struct {
	path string               // served path
	fd   map[uintptr]*os.File // active file descriptors
}

func (f *fileHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL.Path)
	defer r.Body.Close()

	switch r.Method {
	default:
		http.Error(w, "method not allowed: "+r.Method, http.StatusMethodNotAllowed)
	case "OPEN":
		f.open(w, r)
	}
}

func (f *fileHandler) open(w http.ResponseWriter, r *http.Request) {

	// by cleaning the (absolute) path, we sandbox it so that ../ can't go above the root export.
	p := path.Clean(r.URL.Path)
	assert(path.IsAbs(p))
	fname := path.Join(f.path, p)

	// parse open flags
	query := r.URL.Query()
	flagStr := query.Get("flag")
	flag, eFlag := strconv.Atoi(flagStr)
	if eFlag != nil {
		http.Error(w, "invalid flag: "+flagStr, http.StatusBadRequest)
		return
	}

	// parse permissions
	permStr := query.Get("perm")
	perm, ePerm := strconv.Atoi(permStr)
	if ePerm != nil {
		http.Error(w, "invalid perm: "+permStr, http.StatusBadRequest)
		return
	}

	// open file, answer with file descriptor
	file, err := os.OpenFile(fname, flag, os.FileMode(perm))
	if err != nil {
		http.Error(w, err.Error(), 400) // TODO: could distinguish: not found, forbidden, ...
		return
	}
	fd := file.Fd()
	f.fd[fd] = file
	fmt.Fprint(w, fd)
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}

//func(f*fileHandler) get(){
//f, err := os.Open(fname)
//if err != nil {
//	log.Println(err)
//	http.Error(w, err.Error(), http.StatusNotFound) // TODO: others?
//	return
//}
//defer f.Close()
//n, err2 := io.Copy(w, f)
//if err2 != nil {
//	log.Println("upload", fname, ":", err2.Error())
//}
//log.Println(n, "bytes sent")
//}
