rm pcie.txt
for n in 8 16 24 32 48 64 96 128 196 256; do
	echo
	echo n=$n
	echo
	go run pcie.go -warp $(( $n*$n*$n )) $n >> pcie.txt
done;
./pcie.gplot


