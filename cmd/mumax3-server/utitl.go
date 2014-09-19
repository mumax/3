package main

import (
	"log"
	"strings"
	"time"
)

// BaseDir returns the first path element, without slashes and ignoring http:// . E.g.:
// 	/home/user/file -> home
// 	user/file -> user
// 	http://home/user/file -> home
func BaseDir(dir string) string {
	if strings.HasPrefix(dir, "http://") {
		return BaseDir(dir[len("http://"):])
	}
	firstSlash := strings.Index(dir, "/")
	switch {
	case firstSlash < 0:
		return dir
	case firstSlash == 0:
		return BaseDir(dir[1:])
	default:
		return dir[:firstSlash]
	}
}

func Fatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// rounded up to 1s precission
func Since(a, b time.Time) time.Duration {
	d := a.Sub(b)
	return (d/1e9)*1e9 + 1e9
}
