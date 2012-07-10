package ptx

const KERNMUL = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_00007e11_00000000-9_kernmul.cpp3.i (/tmp/ccBI#.tIydOu)
	//-----------------------------------------------------------

	//-----------------------------------------------------------
	// Options:
	//-----------------------------------------------------------
	//  Target:ptx, ISA:sm_10, Endian:little, Pointer Size:64
	//  -O0	(Optimization level)
	//  -g2	(Debug level)
	//  -m2	(Report advisories)
	//-----------------------------------------------------------

	.file	1	"<command-line>"
	.file	2	"/tmp/tmpxft_00007e11_00000000-8_kernmul.cudafe2.gpu"
	.file	3	"/usr/lib/gcc/x86_64-linux-gnu/4.6/include/stddef.h"
	.file	4	"/usr/local/cuda/bin/../include/crt/device_runtime.h"
	.file	5	"/usr/local/cuda/bin/../include/host_defines.h"
	.file	6	"/usr/local/cuda/bin/../include/builtin_types.h"
	.file	7	"/usr/local/cuda/bin/../include/device_types.h"
	.file	8	"/usr/local/cuda/bin/../include/driver_types.h"
	.file	9	"/usr/local/cuda/bin/../include/surface_types.h"
	.file	10	"/usr/local/cuda/bin/../include/texture_types.h"
	.file	11	"/usr/local/cuda/bin/../include/vector_types.h"
	.file	12	"/usr/local/cuda/bin/../include/device_launch_parameters.h"
	.file	13	"/usr/local/cuda/bin/../include/crt/storage_class.h"
	.file	14	"kernmul.cu"
	.file	15	"/usr/local/cuda/bin/../include/common_functions.h"
	.file	16	"/usr/local/cuda/bin/../include/math_functions.h"
	.file	17	"/usr/local/cuda/bin/../include/math_constants.h"
	.file	18	"/usr/local/cuda/bin/../include/device_functions.h"
	.file	19	"/usr/local/cuda/bin/../include/sm_11_atomic_functions.h"
	.file	20	"/usr/local/cuda/bin/../include/sm_12_atomic_functions.h"
	.file	21	"/usr/local/cuda/bin/../include/sm_13_double_functions.h"
	.file	22	"/usr/local/cuda/bin/../include/sm_20_atomic_functions.h"
	.file	23	"/usr/local/cuda/bin/../include/sm_20_intrinsics.h"
	.file	24	"/usr/local/cuda/bin/../include/sm_30_intrinsics.h"
	.file	25	"/usr/local/cuda/bin/../include/surface_functions.h"
	.file	26	"/usr/local/cuda/bin/../include/texture_fetch_functions.h"
	.file	27	"/usr/local/cuda/bin/../include/math_functions_dbl_ptx1.h"


	.entry kernmul
	{
	.loc	14	3	0
$LDWbegin_kernmul:
	.loc	14	5	0
	exit;
$LDWend_kernmul:
	} // kernmul

 	@@DWARF .section .debug_info, "",@progbits
	@@DWARF .byte	0x95, 0x02, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_abbrev
	@@DWARF .4byte	0x742f0108, 0x742f706d, 0x6678706d, 0x30305f74
	@@DWARF .4byte	0x65373030, 0x305f3131, 0x30303030, 0x2d303030
	@@DWARF .4byte	0x656b5f39, 0x756d6e72, 0x70632e6c, 0x692e3370
	@@DWARF .4byte	0x6f682f00, 0x612f656d, 0x2f656e72, 0x732f6f67
	@@DWARF .4byte	0x6e2f6372, 0x6c626d69, 0x75632d65, 0x702f6562
	@@DWARF .4byte	0x6f007874, 0x636e6570, 0x2e342063, 0x00040031
	@@DWARF .4byte	.debug_line
	@@DWARF .4byte	0x736e7502, 0x656e6769, 0x6e692064, 0x04070074
	@@DWARF .4byte	0x75bb0b03, 0x33746e69, 0x00b40c00, 0x0b040000
	@@DWARF .4byte	0x6e0078bc, 0x02000000, 0x04010023, 0x0079bc0b
	@@DWARF .4byte	0x0000006e, 0x01042302, 0x7abc0b04, 0x00006e00
	@@DWARF .4byte	0x08230200, 0x0b050001, 0x69750167, 0x0033746e
	@@DWARF .4byte	0x0000007e, 0x0000b406, 0x8a0b0700, 0x6d696401
	@@DWARF .4byte	0x000c0033, 0x08000001, 0x78018b0b, 0x00006e00
	@@DWARF .4byte	0x00230200, 0x8b0b0801, 0x6e007901, 0x02000000
	@@DWARF .4byte	0x08010423, 0x7a018b0b, 0x00006e00, 0x08230200
	@@DWARF .4byte	0x0b050001, 0x69640193, 0xc700336d, 0x06000000
	@@DWARF .4byte	0x00000100, 0x746e6902, 0x06040500, 0x00000112
	@@DWARF .4byte	0x6f6c6602, 0x04007461, 0x011e0904, 0x05080000
	@@DWARF .4byte	0x756f6402, 0x00656c62, 0x6c020804, 0x20676e6f
	@@DWARF .4byte	0x676e6f6c, 0x736e7520, 0x656e6769, 0x6e692064
	@@DWARF .4byte	0x08070074, 0x6e6f6c02, 0x6f6c2067, 0x6920676e
	@@DWARF .4byte	0x0500746e, 0x3e070a08, 0x61647563, 0x6e756f52
	@@DWARF .4byte	0x646f4d64, 0xd5040065, 0x0b000001, 0x75633f07
	@@DWARF .4byte	0x6f526164, 0x4e646e75, 0x65726165, 0x00007473
	@@DWARF .4byte	0x0b000000, 0x75634007, 0x6f526164, 0x5a646e75
	@@DWARF .4byte	0x006f7265, 0x00000001, 0x6341070b, 0x52616475
	@@DWARF .4byte	0x646e756f, 0x49736f50, 0x0200666e, 0x0b000000
	@@DWARF .4byte	0x75634307, 0x6f526164, 0x4d646e75, 0x6e496e69
	@@DWARF .4byte	0x00030066, 0x02000000, 0x676e6f6c, 0x746e6920
	@@DWARF .4byte	0x02080500, 0x72616863, 0x06010600, 0x000001e1
	@@DWARF .4byte	0x0001e909, 0x03050800, 0x6c66fd0b, 0x3274616f
	@@DWARF .4byte	0x021f0800, 0x0b040000, 0x1e0078fd, 0x02000001
	@@DWARF .4byte	0x04010023, 0x0079fd0b, 0x0000011e, 0x01042302
	@@DWARF .4byte	0x730b0500, 0x6f6c6601, 0x00327461, 0x000001f5
	@@DWARF .4byte	0x00006e0c, 0x023b0100, 0x050d0000, 0x14100e00
	@@DWARF .4byte	0x635f5f0d, 0x72616475, 0x32695f74, 0x5f69706f
	@@DWARF .4byte	0x022e0066, 0x12090000, 0x08000001, 0x006e0c05
	@@DWARF .4byte	0x68010000, 0x0d000002, 0x1e0f0006, 0x06000001
	@@DWARF .4byte	0x0000011e, 0x00012e09, 0x10050800, 0x656b030e
	@@DWARF .byte	0x72, 0x6e, 0x6d, 0x75, 0x6c, 0x00, 0x01, 0x01
	@@DWARF .byte	0x00
	@@DWARF .quad	$LDWbegin_kernmul
	@@DWARF .quad	$LDWend_kernmul
	@@DWARF .byte	0x00, 0x00

 	@@DWARF .section .debug_pubnames, "",@progbits
	@@DWARF .byte	0x1a, 0x00, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_info
	@@DWARF .4byte	0x00000299, 0x00000279, 0x6e72656b, 0x006c756d
	@@DWARF .byte	0x00, 0x00, 0x00, 0x00

 	@@DWARF .section .debug_abbrev, "",@progbits
	@@DWARF .4byte	0x03011101, 0x25081b08, 0x420b1308, 0x0006100b
	@@DWARF .4byte	0x00240200, 0x0b3e0803, 0x00000b0b, 0x3a011303
	@@DWARF .4byte	0x030b3b0b, 0x010b0b08, 0x04000013, 0x0b3a000d
	@@DWARF .4byte	0x08030b3b, 0x0a381349, 0x00000b32, 0x3a001605
	@@DWARF .4byte	0x03053b0b, 0x00134908, 0x00260600, 0x00001349
	@@DWARF .4byte	0x3a011307, 0x03053b0b, 0x010b0b08, 0x08000013
	@@DWARF .4byte	0x0b3a000d, 0x0803053b, 0x0a381349, 0x00000b32
	@@DWARF .4byte	0x49000f09, 0x330b0b13, 0x0a00000b, 0x0b3a0104
	@@DWARF .4byte	0x08030b3b, 0x13010b0b, 0x280b0000, 0x3b0b3a00
	@@DWARF .4byte	0x1c08030b, 0x0c000006, 0x13490101, 0x13010c3c
	@@DWARF .4byte	0x210d0000, 0x000b2f00, 0x00340e00, 0x053b0b3a
	@@DWARF .4byte	0x13490803, 0x350f0000, 0x00134900, 0x002e1000
	@@DWARF .4byte	0x0b3b0b3a, 0x0c3f0803, 0x0a400c27, 0x01120111
	@@DWARF .byte	0x00, 0x00, 0x00, 0x00

`
