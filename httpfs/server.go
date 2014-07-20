package httpfs

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"sync"
)

const (
	X_READ_ERROR = "X-HTTPFS-ReadError"
)

type server struct {
	path string // served path
	sync.Mutex
	openFiles map[uintptr]*os.File // active file descriptors
}

func Serve(root, addr string) error {
	log.Println("serving", root, "at", addr)
	server := &server{path: root, openFiles: make(map[uintptr]*os.File)}
	err := http.ListenAndServe(addr, server) // don't use DefaultServeMux which redirects some requests behind our back.
	return err
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL)
	defer r.Body.Close()

	switch r.Method {
	default:
		msg := "method not allowed: " + r.Method
		log.Println("ERROR:", msg)
		http.Error(w, msg, http.StatusMethodNotAllowed)
	case "OPEN":
		s.open(w, r)
	case "READ":
		s.read(w, r)
	case "WRITE":
		s.write(w, r)
	}
}

func (s *server) open(w http.ResponseWriter, r *http.Request) {
	// by cleaning the (absolute) path, we sandbox it so that ../ can't go above the root export.
	p := path.Clean(r.URL.Path)
	assert(path.IsAbs(p))
	fname := path.Join(s.path, p)

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
	perm, ePerm := strconv.Atoi(permStr) // TODO: base8 (also client)
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
	s.storeFD(fd, file)
	log.Println("httpfs: opened", fname, ", fd:", fd)
	fmt.Fprint(w, fd) // respond with file descriptor
}

func (s *server) read(w http.ResponseWriter, r *http.Request) {
	// TODO: limit N
	fdStr := r.URL.Path[len("/"):]
	fd, _ := strconv.Atoi(fdStr)
	file := s.getFD(uintptr(fd))
	if file == nil {
		http.Error(w, "read: invalid file descriptor: "+fdStr, http.StatusBadRequest)
		return
	}

	nStr := r.URL.Query().Get("n")
	n, eN := strconv.Atoi(nStr)
	if eN != nil {
		http.Error(w, "read: invalid number of bytes: "+nStr, http.StatusBadRequest)
		return
	}

	log.Println("httpfs: read fd", fd, "n=", n)

	// first read into buffer...
	buf := make([]byte, n)
	nRead, err := file.Read(buf)
	// ...so we can put error in header
	if err != nil && err != io.EOF {
		w.Header().Set(X_READ_ERROR, err.Error())
	}
	// upload error is server error, not client. TODO: client: check if enough received!
	nUpload, eUpload := w.Write(buf[:nRead])
	log.Println(nUpload, "bytes uploaded, error=", eUpload)
}

func (s *server) write(w http.ResponseWriter, r *http.Request) {
	fdStr := r.URL.Path[len("/"):]
	fd, _ := strconv.Atoi(fdStr)
	file := s.getFD(uintptr(fd))
	if file == nil {
		http.Error(w, "read: invalid file descriptor: "+fdStr, http.StatusBadRequest)
		return
	}

	n, err := io.Copy(file, r.Body)
	fmt.Fprint(w, n, err)
}

// thread-safe openFiles[fd]
func (s *server) getFD(fd uintptr) *os.File {
	s.Lock()
	defer s.Unlock()
	return s.openFiles[fd] // TODO: protect against nonexisting FD
}

// thread-safe openFiles[fd] = f
func (s *server) storeFD(fd uintptr, f *os.File) {
	s.Lock()
	defer s.Unlock()
	s.openFiles[fd] = f
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
