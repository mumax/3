/*
mumax3-format uses the standard Go formatter (`gofmt`) to standardize
spacing, line endings etc. of mumax3 input files.
(NOTE: only files with the `.mx3` extension are considered.)
This can, in some cases, be used to fix corrupt input files.

# Usage

Format a single mumax3 input file:

	mumax3-format file.mx3

Format all mumax3 input files (.mx3) in a directory:

	mumax3-format directory_path

*/

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"go/format"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	// Get the directory path from the command-line arguments
	if len(os.Args) < 2 {
		fmt.Println("Usage: mumax3-format <file_or_folder>")
		return
	}
	folder := os.Args[1]
	info, err := os.Stat(folder)
	if err != nil {
		fmt.Printf("Error finding %s: %v\n", folder, err)
		return
	}

	// Walk through the directory and find all .mx3 files
	err = filepath.WalkDir(folder, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Check if the file has a .mx3 extension
		if !d.IsDir() && (filepath.Ext(path) == ".mx3" || !info.IsDir()) {
			// Read the .mx3 file
			source, err := os.ReadFile(path)
			if err != nil {
				return fmt.Errorf("failed to read file %s: %w", path, err)
			}

			// Split file into lines to process each line individually
			scanner := bufio.NewScanner(bytes.NewReader(source))
			var buffer bytes.Buffer
			var codeBuffer bytes.Buffer
			inCodeBlock := false

			for scanner.Scan() {
				line := scanner.Text()

				// Check if line is a comment
				if strings.HasPrefix(line, "//") {
					// Flush the code buffer before writing the comment
					if inCodeBlock {
						formattedCode, err := formatCodeBuffer(&codeBuffer)
						if err != nil {
							buffer.WriteString(codeBuffer.String()) // Write as-is if formatting fails
						} else {
							buffer.Write(formattedCode)
						}
						codeBuffer.Reset()
						inCodeBlock = false
					}
					// Write comment line as-is
					buffer.WriteString(line + "\n")
				} else {
					// Accumulate lines of code
					codeBuffer.WriteString(line + "\n")
					inCodeBlock = true
				}
			}

			// Flush any remaining code in the buffer
			if inCodeBlock {
				formattedCode, err := formatCodeBuffer(&codeBuffer)
				if err != nil {
					buffer.WriteString(codeBuffer.String()) // Write as-is if formatting fails
				} else {
					buffer.Write(formattedCode)
				}
			}

			// Check for scanner error
			if err := scanner.Err(); err != nil {
				return fmt.Errorf("failed to scan file %s: %w", path, err)
			}

			// Write the processed content back to the file
			err = os.WriteFile(path, buffer.Bytes(), 0644)
			if err != nil {
				return fmt.Errorf("failed to write file %s: %w", path, err)
			}

			fmt.Printf("File formatted: %s\n", path)
		}
		return nil
	})

	if err != nil {
		fmt.Printf("Error walking through directory: %v\n", err)
	}
}

// Helper function to format a block of code
func formatCodeBuffer(codeBuffer *bytes.Buffer) ([]byte, error) {
	return format.Source(codeBuffer.Bytes())
}
