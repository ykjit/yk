Notes
- I hardcoded the margin between the offset and the desired destination -  0x0D = 13
- I can return from one control point to the other

 0x0000000000202654  main+356 mov    0x204a80,%rdi
 0x000000000020265c  main+364 mov    0x0(%r13),%edx
 0x0000000000202660  main+368 movabs $0x201eda,%rsi
 0x000000000020266a  main+378 mov    $0x0,%al
 0x000000000020266c  main+380 call   0x202800 <fprintf@plt>
 0x0000000000202671  main+385 jmp    0x202673 <main+387>

>>> info symbol 0x0000000000202643
__yk_clone_main + 339 in section .text of /tmp/.tmpqchUcw/simple2


- I need to find the correct place to insert my pass after the optimisation are done but before the IR is compiled.
- 

>>> info symbol 0x2025e3
__yk_clone_main + 339 in section .text of /tmp/.tmpxa8iPl/simple2


disassemble 0x2025c6
   0x00000000002025ab <+299>:	mov    $0x9,%edi
   0x00000000002025b0 <+304>:	mov    $0x9,%esi
   0x00000000002025b5 <+309>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002025ba <+314>:	mov    -0x30(%rbp),%rdi
   0x00000000002025be <+318>:	mov    %rbx,%rsi
   0x00000000002025c1 <+321>:	mov    $0x1,%edx
   0x00000000002025c6 <+326>:	movabs $0x202740,%r11 <----------------------------
   0x00000000002025d0 <+336>:	call   *%r11
   0x00000000002025d3 <+339>:	jmp    0x2025d5 <main+341>
   0x00000000002025d5 <+341>:	mov    $0x9,%edi
   0x00000000002025da <+346>:	mov    $0xa,%esi
   0x00000000002025df <+351>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002025e4 <+356>:	mov    0x204a00,%rdi
   0x00000000002025ec <+364>:	mov    0x0(%r13),%edx
   0x00000000002025f0 <+368>:	movabs $0x201e6a,%rsi


disassemble 0x202740

This it where we return (based on rec.offset)
0x2025c6


 > 0x7ffff7fbc001  movabs $0x2025c6,%rsp


When obtaining the record offset I am returned to 0x2025c6:

 0x2025c1 <main+321>     mov    $0x1,%edx
 0x2025c6 <main+326>     movabs $0x202740,%r11 
 0x2025d0 <main+336>     call   *%r11
 0x2025d3 <main+339>     jmp    0x2025d5 <main+341>

But I actually want to be at 0x2025c6.



0x202384 -> tracing calls

   0x0000000000202376 <+294>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020237b <+299>:	mov    -0x30(%rbp),%rdi
   0x000000000020237f <+303>:	mov    %rbx,%rsi
   0x0000000000202382 <+306>:	xor    %edx,%edx
   0x0000000000202384 <+308>:	movabs $0x202740,%r11


0x2025c6 -> 

   0x00000000002025b5 <+309>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002025ba <+314>:	mov    -0x30(%rbp),%rdi
   0x00000000002025be <+318>:	mov    %rbx,%rsi
   0x00000000002025c1 <+321>:	mov    $0x1,%edx
   0x00000000002025c6 <+326>:	movabs $0x202740,%r11



>>> info symbol 0x2025d6
__yk_clone_main + 326 in section .text of /tmp/.tmpBLAa9m/simple2


>>> info symbol 0x202384
main + 292 in section .text of /tmp/.tmpLawIlS/simple2

@@@@@@@@@ OPT SM ID 0, offset: 0x202384, target_addr: 0x202391
@@@@@@@@@ OPT SM ID 1, offset: 0x2025c6, target_addr: 0x2025d3

>>> info symbol 0x2025c6
__yk_clone_main + 310 in section .text of /tmp/.tmpLawIlS/simple2


>>> disassemble 0x2025c6,
Dump of assembler code for function main:
   0x0000000000202480 <+0>:	push   %rbp
   0x0000000000202481 <+1>:	mov    %rsp,%rbp
   0x0000000000202484 <+4>:	push   %r15
   0x0000000000202486 <+6>:	push   %r14
   0x0000000000202488 <+8>:	push   %r13
   0x000000000020248a <+10>:	push   %r12
   0x000000000020248c <+12>:	push   %rbx
   0x000000000020248d <+13>:	sub    $0x18,%rsp
   0x0000000000202491 <+17>:	mov    %rsi,-0x38(%rbp)
   0x0000000000202495 <+21>:	mov    %edi,%r15d
   0x0000000000202498 <+24>:	mov    $0x9,%edi
   0x000000000020249d <+29>:	xor    %esi,%esi
   0x000000000020249f <+31>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002024a4 <+36>:	lea    0x253d(%rip),%rax        # 0x2049e8 <shadowstack_0>
   0x00000000002024ab <+43>:	mov    (%rax),%rax
   0x00000000002024ae <+46>:	mov    %rax,%r14
   0x00000000002024b1 <+49>:	add    $0x8,%r14
   0x00000000002024b5 <+53>:	mov    %rax,%rbx
   0x00000000002024b8 <+56>:	add    $0x10,%rbx
   0x00000000002024bc <+60>:	mov    %rax,%r12
   0x00000000002024bf <+63>:	add    $0x18,%r12
   0x00000000002024c3 <+67>:	mov    %rax,%r13
   0x00000000002024c6 <+70>:	add    $0x1c,%r13
   0x00000000002024ca <+74>:	movl   $0x0,(%rax)
   0x00000000002024d0 <+80>:	mov    %r15d,0x4(%rax)
   0x00000000002024d4 <+84>:	jmp    0x2024d6 <main+86>
   0x00000000002024d6 <+86>:	mov    $0x9,%edi
   0x00000000002024db <+91>:	mov    $0x1,%esi
   0x00000000002024e0 <+96>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002024e5 <+101>:	mov    -0x38(%rbp),%rax
   0x00000000002024e9 <+105>:	mov    %rax,(%r14)
   0x00000000002024ec <+108>:	xor    %edi,%edi
   0x00000000002024ee <+110>:	call   0x202730 <yk_mt_new@plt>
   0x00000000002024f3 <+115>:	mov    %rax,%r14
   0x00000000002024f6 <+118>:	jmp    0x2024f8 <main+120>
   0x00000000002024f8 <+120>:	mov    $0x9,%edi
   0x00000000002024fd <+125>:	mov    $0x2,%esi
   0x0000000000202502 <+130>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202507 <+135>:	mov    %r14,-0x30(%rbp)
   0x000000000020250b <+139>:	mov    -0x30(%rbp),%rdi
   0x000000000020250f <+143>:	xor    %esi,%esi
   0x0000000000202511 <+145>:	call   0x202720 <yk_mt_hot_threshold_set@plt>
   0x0000000000202516 <+150>:	jmp    0x202518 <main+152>
   0x0000000000202518 <+152>:	mov    $0x9,%edi
   0x000000000020251d <+157>:	mov    $0x3,%esi
   0x0000000000202522 <+162>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202527 <+167>:	call   0x202760 <yk_location_new@plt>
   0x000000000020252c <+172>:	mov    %rax,%r14
   0x000000000020252f <+175>:	jmp    0x202531 <main+177>
   0x0000000000202531 <+177>:	mov    $0x9,%edi
   0x0000000000202536 <+182>:	mov    $0x4,%esi
   0x000000000020253b <+187>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202540 <+192>:	mov    %r14,(%rbx)
   0x0000000000202543 <+195>:	movl   $0x270e,(%r12)
   0x000000000020254b <+203>:	movl   $0x4,0x0(%r13)
   0x0000000000202553 <+211>:	mov    (%rbx),%rax0x2025c6
   0x000000000020256b <+235>:	jmp    0x20256d <main+237>
   0x000000000020256d <+237>:	mov    $0x9,%edi
   0x0000000000202572 <+242>:	mov    $0x6,%esi
   0x0000000000202577 <+247>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020257c <+252>:	mov    0x0(%r13),%eax
   0x0000000000202580 <+256>:	jmp    0x202582 <main+258>
   0x0000000000202582 <+258>:	mov    $0x9,%edi
   0x0000000000202587 <+263>:	mov    $0x7,%esi
   0x000000000020258c <+268>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202591 <+273>:	jmp    0x202593 <main+275>
   0x0000000000202593 <+275>:	mov    $0x9,%edi
   0x0000000000202598 <+280>:	mov    $0x8,%esi
   0x000000000020259d <+285>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002025a2 <+290>:	cmpl   $0x0,0x0(%r13)
   0x00000000002025a7 <+295>:	jle    0x202622 <main+418>
   0x00000000002025a9 <+297>:	jmp    0x2025ab <main+299>
   0x00000000002025ab <+299>:	mov    $0x9,%edi
   0x00000000002025b0 <+304>:	mov    $0x9,%esi
   0x00000000002025b5 <+309>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002025ba <+314>:	mov    -0x30(%rbp),%rdi
   0x00000000002025be <+318>:	mov    %rbx,%rsi
   0x00000000002025c1 <+321>:	mov    $0x1,%edx
   0x00000000002025c6 <+326>:	movabs $0x202740,%r11
   0x00000000002025d0 <+336>:	call   *%r11
   0x00000000002025d3 <+339>:	jmp    0x2025d5 <main+341>
   0x00000000002025d5 <+341>:	mov    $0x9,%edi
   0x00000000002025da <+346>:	mov    $0xa,%esi2025d3
   0x00000000002025df <+351>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002025e4 <+356>:	mov    0x204a00,%rdi
   0x00000000002025ec <+364>:	mov    0x0(%r13),%edx
   0x00000000002025f0 <+368>:	movabs $0x201e6a,%rsi
   0x00000000002025fa <+378>:	mov    $0x0,%al
   0x00000000002025fc <+380>:	call   0x202780 <fprintf@plt>
   0x0000000000202601 <+385>:	jmp    0x202603 <main+387>
   0x0000000000202603 <+387>:	mov    $0x9,%edi
   0x0000000000202608 <+392>:	mov    $0xb,%esi
   0x000000000020260d <+397>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202612 <+402>:	mov    0x0(%r13),%eax
   0x0000000000202616 <+406>:	add    $0xffffffff,%eax
   0x0000000000202619 <+409>:	mov    %eax,0x0(%r13)
   0x000000000020261d <+413>:	jmp    0x202593 <main+275>
   0x0000000000202622 <+418>:	mov    $0x9,%edi
   0x0000000000202627 <+423>:	mov    $0xc,%esi
   0x000000000020262c <+428>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202631 <+433>:	mov    0x204a00,%rdi
   0x0000000000202639 <+441>:	movabs $0x201e64,%rsi
   0x0000000000202643 <+451>:	mov    $0x0,%al
   0x0000000000202645 <+453>:	call   0x202780 <fprintf@plt>
   0x000000000020264a <+458>:	jmp    0x20264c <main+460>
   0x000000000020264c <+460>:	mov    $0x9,%edi
   0x0000000000202651 <+465>:	mov    $0xd,%esi
   0x0000000000202656 <+470>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020265b <+475>:	mov    (%r12),%eax
   0x000000000020265f <+479>:	jmp    0x202661 <main+481>
   0x0000000000202661 <+481>:	mov    $0x9,%edi
   0x0000000000202666 <+486>:	mov    $0xe,%esi
   0x000000000020266b <+491>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202670 <+496>:	mov    (%rbx),%rdi
   0x0000000000202673 <+499>:	call   0x202750 <yk_location_drop@plt>
   0x0000000000202678 <+504>:	jmp    0x20267a <main+506>
   0x000000000020267a <+506>:	mov    $0x9,%edi
   0x000000000020267f <+511>:	mov    $0xf,%esi
   0x0000000000202684 <+516>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202689 <+521>:	mov    -0x30(%rbp),%rdi
   0x000000000020268d <+525>:	call   0x202710 <yk_mt_shutdown@plt>
   0x0000000000202692 <+530>:	jmp    0x202694 <main+532>
   0x0000000000202694 <+532>:	mov    $0x9,%edi
   0x0000000000202699 <+537>:	mov    $0x10,%esi
   0x000000000020269e <+542>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002026a3 <+547>:	jmp    0x2026c5 <main+581>
   0x00000000002026a5 <+549>:	mov    $0x9,%edi
   0x00000000002026aa <+554>:	mov    $0x11,%esi
   0x00000000002026af <+559>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002026b4 <+564>:	xor    %eax,%eax
   0x00000000002026b6 <+566>:	add    $0x18,%rsp
   0x00000000002026ba <+570>:	pop    %rbx
   0x00000000002026bb <+571>:	pop    %r12
   0x00000000002026bd <+573>:	pop    %r13
   0x00000000002026bf <+575>:	pop    %r14
   0x00000000002026c1 <+577>:	pop    %r15
   0x00000000002026c3 <+579>:	pop    %rbp
   0x00000000002026c4 <+580>:	ret
   0x00000000002026c5 <+581>:	mov    $0x9,%edi
   0x00000000002026ca <+586>:	mov    $0x12,%esi
   0x00000000002026cf <+591>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002026d4 <+596>:	jmp    0x2026a5 <main+549>
End of assembler dump.
>>> disassemble 0x202384
Dump of assembler code for function main:
   0x0000000000202250 <+0>:	push   %rbp
   0x0000000000202251 <+1>:	mov    %rsp,%rbp
   0x0000000000202254 <+4>:	push   %r15
   0x0000000000202256 <+6>:	push   %r14
   0x0000000000202258 <+8>:	push   %r13
   0x000000000020225a <+10>:	push   %r12
   0x000000000020225c <+12>:	push   %rbx
   0x000000000020225d <+13>:	sub    $0x18,%rsp
   0x0000000000202261 <+17>:	mov    %rsi,-0x38(%rbp)
   0x0000000000202265 <+21>:	mov    %edi,%r15d
   0x0000000000202268 <+24>:	xor    %edi,%edi
   0x000000000020226a <+26>:	xor    %esi,%esi
   0x000000000020226c <+28>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202271 <+33>:	mov    $0xf4240,%edi
   0x0000000000202276 <+38>:	call   0x202790 <malloc@plt>
   0x000000000020227b <+43>:	lea    0x2766(%rip),%rcx        # 0x2049e8 <shadowstack_0>
   0x0000000000202282 <+50>:	mov    %rax,(%rcx)
   0x0000000000202285 <+53>:	mov    %rax,%r14
   0x0000000000202288 <+56>:	add    $0x8,%r14
   0x000000000020228c <+60>:	mov    %rax,%rbx
   0x000000000020228f <+63>:	add    $0x10,%rbx
   0x0000000000202293 <+67>:	mov    %rax,%r12
   0x0000000000202296 <+70>:	add    $0x18,%r12
   0x000000000020229a <+74>:	mov    %rax,%r13
   0x000000000020229d <+77>:	add    $0x1c,%r13
   0x00000000002022a1 <+81>:	movl   $0x0,(%rax)
   0x00000000002022a7 <+87>:	mov    %r15d,0x4(%rax)
   0x00000000002022ab <+91>:	jmp    0x2022ad <main+93>
   0x00000000002022ad <+93>:	xor    %edi,%edi
   0x00000000002022af <+95>:	mov    $0x1,%esi
   0x00000000002022b4 <+100>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002022b9 <+105>:	mov    -0x38(%rbp),%rax
   0x00000000002022bd <+109>:	mov    %rax,(%r14)
   0x00000000002022c0 <+112>:	xor    %edi,%edi
   0x00000000002022c2 <+114>:	call   0x202730 <yk_mt_new@plt>
   0x00000000002022c7 <+119>:	mov    %rax,%r14
   0x00000000002022ca <+122>:	jmp    0x2022cc <main+124>
   0x00000000002022cc <+124>:	xor    %edi,%edi
   0x00000000002022ce <+126>:	mov    $0x2,%esi
   0x00000000002022d3 <+131>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002022d8 <+136>:	mov    %r14,-0x30(%rbp)
   0x00000000002022dc <+140>:	mov    -0x30(%rbp),%rdi
   0x00000000002022e0 <+144>:	xor    %esi,%esi
   0x00000000002022e2 <+146>:	call   0x202720 <yk_mt_hot_threshold_set@plt>
   0x00000000002022e7 <+151>:	jmp    0x2022e9 <main+153>
   0x00000000002022e9 <+153>:	xor    %edi,%edi
   0x00000000002022eb <+155>:	mov    $0x3,%esi
   0x00000000002022f0 <+160>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002022f5 <+165>:	call   0x202760 <yk_location_new@plt>
   0x00000000002022fa <+170>:	mov    %rax,%r14
   0x00000000002022fd <+173>:	jmp    0x2022ff <main+175>
   0x00000000002022ff <+175>:	xor    %edi,%edi
   0x0000000000202301 <+177>:	mov    $0x4,%esi
   0x0000000000202306 <+182>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020230b <+187>:	mov    %r14,(%rbx)
   0x000000000020230e <+190>:	movl   $0x270e,(%r12)
   0x0000000000202316 <+198>:	movl   $0x4,0x0(%r13)
   0x000000000020231e <+206>:	mov    (%rbx),%rax
   0x0000000000202321 <+209>:	jmp    0x202323 <main+211>
   0x0000000000202323 <+211>:	xor    %edi,%edi
   0x0000000000202325 <+213>:	mov    $0x5,%esi
   0x000000000020232a <+218>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020232f <+223>:	mov    (%r12),%eax
   0x0000000000202333 <+227>:	jmp    0x202335 <main+229>
   0x0000000000202335 <+229>:	xor    %edi,%edi
   0x0000000000202337 <+231>:	mov    $0x6,%esi
   0x000000000020233c <+236>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202341 <+241>:	mov    0x0(%r13),%eax
   0x0000000000202345 <+245>:	jmp    0x202347 <main+247>
   0x0000000000202347 <+247>:	xor    %edi,%edi
   0x0000000000202349 <+249>:	mov    $0x7,%esi
   0x000000000020234e <+254>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202353 <+259>:	jmp    0x202355 <main+261>
   0x0000000000202355 <+261>:	xor    %edi,%edi
   0x0000000000202357 <+263>:	mov    $0x8,%esi
   0x000000000020235c <+268>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202361 <+273>:	cmpl   $0x0,0x0(%r13)
   0x0000000000202366 <+278>:	setg   %al
   0x0000000000202369 <+281>:	test   $0x1,%al
   0x000000000020236b <+283>:	jne    0x20236f <main+287>
   0x000000000020236d <+285>:	jmp    0x2023da <main+394>
   0x000000000020236f <+287>:	xor    %edi,%edi
   0x0000000000202371 <+289>:	mov    $0x9,%esi
   0x0000000000202376 <+294>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020237b <+299>:	mov    -0x30(%rbp),%rdi
   0x000000000020237f <+303>:	mov    %rbx,%rsi
   0x0000000000202382 <+306>:	xor    %edx,%edx
   0x0000000000202384 <+308>:	movabs $0x202740,%r11
   0x000000000020238e <+318>:	call   *%r11
   0x0000000000202391 <+321>:	jmp    0x202393 <main+323>
   0x0000000000202393 <+323>:	xor    %edi,%edi
   0x0000000000202395 <+325>:	mov    $0xa,%esi
   0x000000000020239a <+330>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020239f <+335>:	mov    0x204a00,%rdi
   0x00000000002023a7 <+343>:	mov    0x0(%r13),%edx
   0x00000000002023ab <+347>:	movabs $0x201e6a,%rsi
   0x00000000002023b5 <+357>:	mov    $0x0,%al
   0x00000000002023b7 <+359>:	call   0x202780 <fprintf@plt>
   0x00000000002023bc <+364>:	jmp    0x2023be <main+366>
   0x00000000002023be <+366>:	xor    %edi,%edi
   0x00000000002023c0 <+368>:	mov    $0xb,%esi
   0x00000000002023c5 <+373>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002023ca <+378>:	mov    0x0(%r13),%eax
   0x00000000002023ce <+382>:	add    $0xffffffff,%eax
   0x00000000002023d1 <+385>:	mov    %eax,0x0(%r13)
   0x00000000002023d5 <+389>:	jmp    0x202355 <main+261>
   0x00000000002023da <+394>:	xor    %edi,%edi
   0x00000000002023dc <+396>:	mov    $0xc,%esi
   0x00000000002023e1 <+401>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x00000000002023e6 <+406>:	mov    0x204a00,%rdi
   0x00000000002023ee <+414>:	movabs $0x201e64,%rsi
   0x00000000002023f8 <+424>:	mov    $0x0,%al
   0x00000000002023fa <+426>:	call   0x202780 <fprintf@plt>
   0x00000000002023ff <+431>:	jmp    0x202401 <main+433>
   0x0000000000202401 <+433>:	xor    %edi,%edi
   0x0000000000202403 <+435>:	mov    $0xd,%esi
   0x0000000000202408 <+440>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020240d <+445>:	mov    (%r12),%eax
   0x0000000000202411 <+449>:	jmp    0x202413 <main+451>
   0x0000000000202413 <+451>:	xor    %edi,%edi
   0x0000000000202415 <+453>:	mov    $0xe,%esi
   0x000000000020241a <+458>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020241f <+463>:	mov    (%rbx),%rdi
   0x0000000000202422 <+466>:	call   0x202750 <yk_location_drop@plt>
   0x0000000000202427 <+471>:	jmp    0x202429 <main+473>
   0x0000000000202429 <+473>:	xor    %edi,%edi
   0x000000000020242b <+475>:	mov    $0xf,%esi
   0x0000000000202430 <+480>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202435 <+485>:	mov    -0x30(%rbp),%rdi
   0x0000000000202439 <+489>:	call   0x202710 <yk_mt_shutdown@plt>
   0x000000000020243e <+494>:	jmp    0x202440 <main+496>
   0x0000000000202440 <+496>:	xor    %edi,%edi
   0x0000000000202442 <+498>:	mov    $0x10,%esi
   0x0000000000202447 <+503>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020244c <+508>:	jmp    0x20246b <main+539>
   0x000000000020244e <+510>:	xor    %edi,%edi
   0x0000000000202450 <+512>:	mov    $0x11,%esi
   0x0000000000202455 <+517>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x000000000020245a <+522>:	xor    %eax,%eax
   0x000000000020245c <+524>:	add    $0x18,%rsp
   0x0000000000202460 <+528>:	pop    %rbx
   0x0000000000202461 <+529>:	pop    %r12
   0x0000000000202463 <+531>:	pop    %r13
   0x0000000000202465 <+533>:	pop    %r14
   0x0000000000202467 <+535>:	pop    %r15
   0x0000000000202469 <+537>:	pop    %rbp
   0x000000000020246a <+538>:	ret
   0x000000000020246b <+539>:	xor    %edi,%edi
   0x000000000020246d <+541>:	mov    $0x12,%esi
   0x0000000000202472 <+546>:	call   0x202770 <__yk_trace_basicblock@plt>
   0x0000000000202477 <+551>:	jmp    0x20244e <main+510>
