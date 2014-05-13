// package util provides common utilities for all other packages.
package util

import "path"

// Remove extension from file name.
func NoExt(file string) string {
	ext := path.Ext(file)
	return file[:len(file)-len(ext)]
}
