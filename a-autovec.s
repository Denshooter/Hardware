	.file	"autovec.c"
	.intel_syntax noprefix
	.text
	.p2align 4
	.globl	test
	.type	test, @function
test:
.LFB39:
	.cfi_startproc
	endbr64
	xor	eax, eax
	.p2align 4,,10
	.p2align 3
.L2:
	movupd	xmm1, XMMWORD PTR [rdi+rax]
	movupd	xmm0, XMMWORD PTR [rsi+rax]
	addpd	xmm0, xmm1
	movups	XMMWORD PTR [rdi+rax], xmm0
	add	rax, 16
	cmp	rax, 524288
	jne	.L2
	ret
	.cfi_endproc
.LFE39:
	.size	test, .-test
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB40:
	.cfi_startproc
	endbr64
	lea	rax, vec1[rip]
	lea	rdx, vec2[rip]
	movdqa	xmm1, XMMWORD PTR .LC0[rip]
	movdqa	xmm3, XMMWORD PTR .LC1[rip]
	lea	rdi, 524288[rax]
	mov	rsi, rdx
	mov	rcx, rax
	.p2align 4,,10
	.p2align 3
.L6:
	movdqa	xmm0, xmm1
	add	rcx, 32
	paddd	xmm1, xmm3
	add	rsi, 32
	cvtdq2pd	xmm2, xmm0
	pshufd	xmm0, xmm0, 238
	movaps	XMMWORD PTR -32[rcx], xmm2
	cvtdq2pd	xmm0, xmm0
	movaps	XMMWORD PTR -16[rcx], xmm0
	movaps	XMMWORD PTR -32[rsi], xmm2
	movaps	XMMWORD PTR -16[rsi], xmm0
	cmp	rdi, rcx
	jne	.L6
	lea	rcx, vec2[rip+524288]
	.p2align 4,,10
	.p2align 3
.L7:
	movapd	xmm0, XMMWORD PTR [rax]
	addpd	xmm0, XMMWORD PTR [rdx]
	add	rdx, 16
	add	rax, 16
	movaps	XMMWORD PTR -16[rax], xmm0
	cmp	rcx, rdx
	jne	.L7
	xor	eax, eax
	ret
	.cfi_endproc
.LFE40:
	.size	main, .-main
	.globl	vec2
	.bss
	.align 16
	.type	vec2, @object
	.size	vec2, 524288
vec2:
	.zero	524288
	.globl	vec1
	.align 16
	.type	vec1, @object
	.size	vec1, 524288
vec1:
	.zero	524288
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.long	0
	.long	1
	.long	2
	.long	3
	.align 16
.LC1:
	.long	4
	.long	4
	.long	4
	.long	4
	.ident	"GCC: (Ubuntu 12.2.0-17ubuntu1) 12.2.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
