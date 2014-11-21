/*
mumax3-fft performs a Fourier transform on mumax3 table output. E.g.:
 	mumax3-fft table.txt
will create table_fft.txt with per-column FFTs of the data in table.txt.
The first column will contain frequencies in Hz.


Flags


To see all flags, run:
	mumax3-fft -help


Output


By default, the magnitude of the FFT is output. To output magnitude and phase:
 	mumax3-fft -mag -ph table.txt

To output real and imaginary part:
 	mumax3-fft -re -im table.txt
Other combinations of flags may be used.



Auto interpolation


For performance reasons, mumax3's AutoSave outputs data at the requested intervals but truncated to solver time steps. This causes some jitter on the output times, which degrades the FFT spectrum. Therefore mumax3-fft automatically interpolates the data to fit equidistant times. The resulting spectrum is very clean. If desired, interpolation may be turned off by the flag -interpolate=false


Zero padding


To apply zero padding to the input data:
 	mumax3-fft -zeropad 2 table.txt
this will zero-pad the input to 2x its original size, thus increasing the apparent frequency resolution by 2x.



Divide by excitation spectrum


A typical use of mumax3-fft is to extract an FMR spectrum after pulsed excitation. In that case, the spectrum should be divided by the spectrum of the excitation, in complex arithmetic. For convienience, mumax3-fft provides a -divcol flag, to divide the data by the FFT of a column (counting from 1, like gnuplot). E.g.:

 	mumax3-fft -divcol 7 table.txt

gives spectra divided by the (complex) spectrum of column 7 of the file (which may contain, e.g., B_ext_z). Division by the excitation allows to extract meaningful phase information and corrects for non-uniform power spectral density of the exctitation (e.g. when exciting with a gaussian instead of sinc).


Windowing



The following windowing functions are provided: boxcar (no windowing), hamming, hann, welch:
 	mumax3-fft -window hann table.txt



Use with gnuplot


mumax3-fft is easy to use with gnuplot. Inside gnuplot, type, e.g.:
 	gnuplot> plot "<mumax3-fft -stdout table.txt"
this will perform the FFT on-the-fly and pipe the output directly to gnuplot.


License

mumax3-fft inherits the GPLv3 from the FFTW bindings at http://github.com/barnex/fftw

*/
package main
