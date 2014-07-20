package httpfs

// A https File implements a subset of os.File's methods.
type File struct {
	client *Client // client provides the file connection
	name   string  // original file name passed to Open
	fd     uintptr // file descriptor on server
}

func (f *File) Read(p []byte) (n int, err error) {
	return 0, nil
}

func (f *File) Write(p []byte) (n int, err error) {
	return 0, nil
}

func (f *File) Close() error {
	return nil
}
