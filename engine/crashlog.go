package engine

// Saves some output to diagnose why the simulations crashed
func Crashlog() {
	M.SaveAs("m_crash.dump")
	B_eff.SaveAs("B_eff_crash.dump")
}
