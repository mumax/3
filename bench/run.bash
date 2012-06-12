go run bench1.go 16 | grep -v '#' >> bench1_16.txt

rm bench1.txt
for n in 1 2 4 8 16 24 32; do
	echo
	echo n=$n
	echo
	go run bench1.go -warp $(( $n*$n*$n/4 )) $n >> bench1.txt
done;
./bench1.gplot
