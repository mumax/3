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

// HTTP header key for storing errors
const X_ERROR = "X-HTTPFS-Error"

type server struct {
	path string // served path
	sync.Mutex
	openFiles map[uintptr]*os.File // active file descriptors
}

// Serve serves the files under directory root at tcp address addr.
func Serve(root, addr string) error {
	log.Println("serving", root, "at", addr)
	server := &server{path: root, openFiles: make(map[uintptr]*os.File)}
	err := http.ListenAndServe(addr, server) // don't use DefaultServeMux which redirects some requests behind our back.
	return err
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
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
	case "CLOSE":
		s.close(w, r)
	case "MKDIR":
		s.mkdir(w, r)
	}
}

// by cleaning the (absolute) path, we sandbox it so that ../ can't go above the root export.
func (s *server) sandboxPath(p string) string {
	p = path.Clean(p)
	assert(path.IsAbs(p))
	return path.Join(s.path, p)
}

func (s *server) open(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL)

	fname := s.sandboxPath(r.URL.Path)

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
		w.Header().Set(X_ERROR, err.Error())
		w.WriteHeader(statusCode(err))
		return
	}
	fd := file.Fd()
	s.storeFD(fd, file)
	log.Println("httpfs: opened", fname, ", fd:", fd)
	fmt.Fprint(w, fd) // respond with file descriptor
}

func (s *server) mkdir(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL)

	fname := s.sandboxPath(r.URL.Path)

	// parse permissions
	query := r.URL.Query()
	permStr := query.Get("perm")
	perm, ePerm := strconv.Atoi(permStr) // TODO: base8 (also client)
	if ePerm != nil {
		http.Error(w, "invalid perm: "+permStr, http.StatusBadRequest)
		return
	}

	err := os.Mkdir(fname, os.FileMode(perm))
	if err != nil {
		w.Header().Set(X_ERROR, err.Error())
		w.WriteHeader(statusCode(err))
		return
	}
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

	// Go http server does not support Trailer,
	// first read into buffer...
	buf := make([]byte, n)
	nRead, err := file.Read(buf)

	// ...so we can put error in header
	if err != nil && err != io.EOF {
		w.Header().Set(X_ERROR, err.Error())
		w.WriteHeader(400)
		return
	}
	if err == io.EOF {
		w.Header().Set(X_ERROR, "EOF")
	}
	// upload error is server error, not client.
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
	if err != nil {
		w.Header().Set(X_ERROR, err.Error())
		w.WriteHeader(400)
		return
	}
	fmt.Fprint(w, n)
}

func (s *server) close(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL)

	fdStr := r.URL.Path[len("/"):]
	fd, _ := strconv.Atoi(fdStr)
	file := s.getFD(uintptr(fd))
	if file == nil {
		w.Header().Set(X_ERROR, "invalid argument")
		w.WriteHeader(400)
		return
	}

	err := file.Close()
	if err != nil {
		w.Header().Set(X_ERROR, err.Error())
		w.WriteHeader(400)
		return
	}
	s.rmFD(uintptr(fd))
}

// return suited status code for error
func statusCode(err error) int {
	switch {
	default:
		return 400 // general error
	case err == nil:
		return http.StatusOK
	case os.IsNotExist(err):
		return http.StatusNotFound
	case os.IsPermission(err):
		return http.StatusForbidden
	case os.IsExist(err):
		return http.StatusFound
	}
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

// thread-safe delete(openFiles,fd)
func (s *server) rmFD(fd uintptr) {
	s.Lock()
	defer s.Unlock()
	delete(s.openFiles, fd)
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
