package engine

func init() {
	// There are no unsafe features since version 3.10, but we want maximal backwards compatibility
	DeclFunc("ext_EnableUnsafe", EnableUnsafe, "Deprecated. Only here to ensure maximal backwards compatibility with mumax3.9c.")
}

func EnableUnsafe() {
}
