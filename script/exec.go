package script

func (w *World) Exec(src string) error {
	code, err := w.Compile(src)
	if err != nil {
		return err
	}
	code.Exec()
	return nil
}

func (w *World) MustExec(src string) {
	code := w.MustCompile(src)
	code.Exec()
}
