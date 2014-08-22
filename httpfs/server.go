package httpfs

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
)

const (
	X_ERROR  = "X-HTTPFS-Error" // HTTP header key for storing errors
	X_HTTPFS = "X-HTTPFS"       // HTTP header identifying response comes from httpfs server, not other
)

type server struct {
	root string // served path
	sync.Mutex
	openFiles map[uintptr]*os.File // active file descriptors
}

// Serve serves the files under directory root at tcp address addr.
func ListenAndServe(root, addr string) error {

	URL, err := url.Parse(addr)
	if err != nil {
		panic(err)
	}
	if URL.Scheme != "http" {
		panic("httpfs: unknown scheme:" + URL.Scheme)
	}

	addr = URL.Host
	prefix := URL.Path

	l, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return Serve(root, l, prefix)
}

func Serve(root string, l net.Listener, prefix string) error {
	server := NewServer(root, prefix)
	err := http.Serve(l, server) // don't use DefaultServeMux which redirects some requests behind our back.
	return err
}

// NewServer returns a http handler that serves the fs rooted at given root directory.
func NewServer(root string, prefix string) http.Handler {
	//	if !path.IsAbs(root) {
	//		panic(`httpfs needs absolute root path, got: "` + root + `"`)
	//	}
	return http.StripPrefix(prefix, &server{root: root, openFiles: make(map[uintptr]*os.File)})
}

var methods = map[string]func(*server, http.ResponseWriter, *http.Request) error{
	"GET":       (*server).get,
	"OPEN":      (*server).open,
	"READ":      (*server).read,
	"WRITE":     (*server).write,
	"CLOSE":     (*server).close,
	"MKDIR":     (*server).mkdir,
	"READDIR":   (*server).readdir,
	"DELETE":    (*server).delete,
	"REMOVEALL": (*server).removeAll,
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	//log.Println("httpfs server:", r.Method, r.URL)
	//defer log.Println("<<httpfs server done:", r.Method, r.URL)

	//time.Sleep(30*time.Millisecond) // artificial latency for benchmarking

	// crash protection
	defer func() {
		if err := recover(); err != nil {
			log.Println("httpfs panic:", err)
			w.Header().Set(X_ERROR, fmt.Sprint("panic: ", err))
			w.WriteHeader(http.StatusInternalServerError)
		}
	}()

	w.Header().Set(X_HTTPFS, "true")
	handler := methods[r.Method]
	if handler == nil {
		w.Header().Set(X_ERROR, "method not allowed: "+r.Method)
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	err := handler(s, w, r)
	if err != nil {
		w.Header().Set(X_ERROR, err.Error())
		w.WriteHeader(statusCode(err))
		log.Println("httpfs server error:", err)
	}
}

func (s *server) get(w http.ResponseWriter, r *http.Request) error {
	info := map[string]interface{}{
		"Type":      "httpfs server",
		"RootDir":   s.root,
		"OpenFiles": s.LsOF()}
	bytes, err := json.MarshalIndent(info, "", "\t")
	if err != nil {
		return err
	}
	_, eWrite := w.Write(bytes)
	return eWrite
}

func (s *server) LsOF() []string {
	s.Lock()
	defer s.Unlock()
	of := make([]string, 0, len(s.openFiles))
	for _, v := range s.openFiles {
		of = append(of, v.Name())
	}
	sort.Strings(of)
	return of
}

func (s *server) NOpenFiles() int {
	s.Lock()
	defer s.Unlock()
	return len(s.openFiles)
}

// by cleaning the (absolute) path, we sandbox it so that ../ can't go above the root export.
func (s *server) sandboxPath(p string) string {
	p = path.Clean("/" + p)
	assert(path.IsAbs(p))
	return path.Join(s.root, p)
}

// open file, respond with file descriptor
func (s *server) open(w http.ResponseWriter, r *http.Request) error {
	fname := s.sandboxPath(r.URL.Path)
	flag := intQuery(r, "flag")
	perm := intQuery(r, "perm")

	file, err := os.OpenFile(fname, flag, os.FileMode(perm))
	if err != nil {
		return err
	}
	fd := file.Fd()
	s.storeFD(fd, file)
	//log.Println("open", fname, "->", fd)
	fmt.Fprint(w, fd) // respond with file descriptor
	return nil
}

func (s *server) mkdir(w http.ResponseWriter, r *http.Request) error {
	fname := s.sandboxPath(r.URL.Path)
	perm := intQuery(r, "perm")
	return os.Mkdir(fname, os.FileMode(perm))
}

func intQuery(r *http.Request, key string) int {
	str := r.URL.Query().Get(key)
	n, err := strconv.Atoi(str)
	if err != nil {
		log.Println(err) //should not happen
		return 0
	}
	return n
}

func (s *server) read(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}
	n := intQuery(r, "n")

	// Go http server does not support Trailer,
	// first read into buffer...
	buf := make([]byte, n)
	nRead, err := file.Read(buf)
	// ...so we can put error in header
	if err != nil && err != io.EOF {
		return err
	}
	if err == io.EOF {
		w.Header().Set(X_ERROR, "EOF")
		// But statusOK, EOF not treated as actual error
	}
	// upload error is server error, not client.
	//log.Println("uploading", nRead, "bytes:", string(buf[:10])+"...")
	_, eUpload := w.Write(buf[:nRead])
	if eUpload != nil {
		log.Println("upload error for read FD", file.Fd(), ":", eUpload)
	}
	return nil
}

func (s *server) write(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}

	n, err := io.Copy(file, r.Body)
	if err != nil {
		return err
	}
	fmt.Fprint(w, n)
	return nil
}

func (s *server) close(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}
	s.rmFD(file.Fd())
	return file.Close()
}

func (s *server) delete(w http.ResponseWriter, r *http.Request) error {
	path := s.sandboxPath(r.URL.Path)
	return os.Remove(path)
}

func (s *server) removeAll(w http.ResponseWriter, r *http.Request) error {
	path := s.sandboxPath(r.URL.Path)
	return os.RemoveAll(path)
}

func (s *server) readdir(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}
	fi, err := file.Readdir(intQuery(r, "n"))
	if err != nil {
		return err
	}
	list := make([]fileInfo, len(fi))
	for i, fi := range fi {
		list[i] = fileInfo{
			Nm:   fi.Name(),
			Sz:   fi.Size(),
			Md:   fi.Mode(),
			MdTm: fi.ModTime(),
		}
	}
	return json.NewEncoder(w).Encode(list)
}

// retrieve file belonging to file descriptor in URL. E.g.:
// 	http://server/2 -> openFiles[2]
func (s *server) parseFD(URL *url.URL) *os.File {
	fdStr := strings.Trim(URL.Path, "/")
	fd, _ := strconv.Atoi(fdStr) // fd == 0 is error
	return s.getFD(uintptr(fd))
}

// return suited status code for error
func statusCode(err error) int {
	switch {
	default:
		return 400 // general error
	case err == nil:
		return http.StatusOK
	case err == io.EOF:
		return http.StatusOK // EOF not treated as real error
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
