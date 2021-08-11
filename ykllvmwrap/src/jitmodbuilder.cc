// Classes and functions for constructing a new LLVM module from a trace.

using namespace llvm;
using namespace std;

// An atomic counter used to issue compiled traces with unique names.
atomic<uint64_t> NextTraceIdx(0);
uint64_t getNewTraceIdx() {
  uint64_t TraceIdx = NextTraceIdx.fetch_add(1, memory_order_relaxed);
  assert(TraceIdx != numeric_limits<uint64_t>::max());
  return TraceIdx;
}

#define TRACE_FUNC_PREFIX "__yk_compiled_trace_"
#define YKTRACE_START "__yktrace_start_tracing"
#define YKTRACE_STOP "__yktrace_stop_tracing"

// Dump an error message and an LLVM value to stderr and exit with failure.
void dumpValueAndExit(const char *Msg, Value *V) {
  errs() << Msg << ": ";
  V->dump();
  exit(EXIT_FAILURE);
}

std::vector<Value *> getTraceInputs(Function *F, uintptr_t BBIdx) {
  std::vector<Value *> Vec;
  auto It = F->begin();
  // Skip to the first block in the trace which contains the `start_tracing`
  // call.
  std::advance(It, BBIdx);
  BasicBlock *BB = &*It;
  bool found = false;
  for (auto I = BB->begin(); I != BB->end(); I++) {
    if (isa<CallInst>(I)) {
      CallInst *CI = cast<CallInst>(&*I);
      Function *CF = CI->getCalledFunction();
      if ((CF != nullptr) && (CF->getName() == YKTRACE_START)) {
        // Skip first argument to start_tracing.
        for (auto Arg = CI->arg_begin() + 1; Arg != CI->arg_end(); Arg++) {
          Vec.push_back(Arg->get());
        }
        found = true;
        break;
      }
    }
  }
  if (!found)
    errx(EXIT_FAILURE, "failed to find trace inputs");
  return Vec;
}

class JITModBuilder {
  // Global variables/functions that were copied over and need to be
  // initialised.
  vector<GlobalVariable *> cloned_globals;
  // The module being traced.
  Module *AOTMod;
  // The new module that is being build.
  Module *JITMod;
  // A pointer to the call to YKTRACE_START in the AOT module (once
  // encountered). When this changes from NULL to non-NULL, then we start
  // copying instructions from the AOT module into the JIT module.
  Instruction *StartTracingInstr = nullptr;
  // Stack of inlined calls, required to resume at the correct place in the
  // caller.
  std::vector<tuple<size_t, CallInst *>> InlinedCalls;
  // Instruction at which to continue after an a call.
  Optional<tuple<size_t, CallInst *>> ResumeAfter;
  // Depth of nested calls when outlining a recursive function.
  size_t RecCallDepth = 0;
  // Signifies a hole (for which we have no IR) in the trace.
  bool ExpectUnmappable = false;
  // The JITMod's builder.
  llvm::IRBuilder<> Builder;

  // Information about the trace we are compiling.
  // FIXME: These should be grouped into structs.
  char **FuncNames;
  size_t *BBs;
  size_t Len;
  char **FAddrKeys;
  uint64_t *FAddrVals;
  size_t FAddrLen;

  Value *getMappedValue(Value *V) {
    if (VMap.find(V) != VMap.end()) {
      return VMap[V];
    }
    assert(isa<Constant>(V));
    return V;
  }

  // Returns true if the given function exists on the call stack, which means
  // this is a recursive call.
  bool isRecursiveCall(Function *F) {
    for (auto Tup : InlinedCalls) {
      CallInst *CInst = get<1>(Tup);
      if (CInst->getCalledFunction() == F) {
        return true;
      }
    }
    return false;
  }

  // Add an external declaration for the given function to JITMod.
  void declareFunction(Function *F) {
    assert(JITMod->getFunction(F->getName()) == nullptr);
    auto DeclFunc = llvm::Function::Create(F->getFunctionType(),
                                           GlobalValue::ExternalLinkage,
                                           F->getName(), JITMod);
    VMap[F] = DeclFunc;
  }

  // Find the machine code corresponding to the given AOT IR function and
  // ensure there's a mapping from its name to that machine code.
  void addGlobalMappingForFunction(Function *CF) {
    StringRef CFName = CF->getName();
    for (size_t i = 0; i < FAddrLen; i++) {
      char *FName = FAddrKeys[i];
      uint64_t FAddr = FAddrVals[i];
      if (strcmp(FName, CFName.data()) == 0) {
        globalMappings.insert(pair<StringRef, uint64_t>(CFName, FAddr));
        break;
      }
    }
  }

  void handleCallInst(CallInst *CI, Function *CF, size_t &CurInstrIdx) {
    if (CF->isDeclaration()) {
      // The definition of the callee is external to AOTMod. We still
      // need to declare it locally if we have not done so yet.
      if (VMap.find(CF) == VMap.end()) {
        declareFunction(CF);
      }
      if (RecCallDepth == 0) {
        copyInstruction(&Builder, (Instruction *)&*CI);
      }
      // We should expect an "unmappable hole" in the trace. This is
      // where the trace followed a call into external code for which we
      // have no IR, and thus we cannot map blocks for.
      ExpectUnmappable = true;
      ResumeAfter = make_tuple(CurInstrIdx, CI);
    } else {
      if (RecCallDepth > 0) {
        // When outlining a recursive function, we need to count all other
        // function calls so we know when we left the recusion.
        RecCallDepth += 1;
        InlinedCalls.push_back(make_tuple(CurInstrIdx, CI));
        return;
      }
      // If this is a recursive call that has been inlined, remove the
      // inlined code and turn it into a normal call.
      if (isRecursiveCall(CF)) {
        if (VMap.find(CF) == VMap.end()) {
          declareFunction(CF);
          addGlobalMappingForFunction(CF);
        }
        copyInstruction(&Builder, CI);
        InlinedCalls.push_back(make_tuple(CurInstrIdx, CI));
        RecCallDepth = 1;
        return;
      }
      // This is neither recursion nor an external call, so keep it inlined.
      InlinedCalls.push_back(make_tuple(CurInstrIdx, CI));
      // Remap function arguments to the variables passed in by the caller.
      for (unsigned int i = 0; i < CI->arg_size(); i++) {
        Value *Var = CI->getArgOperand(i);
        Value *Arg = CF->getArg(i);
        // Check the operand for things we need to remap, e.g. globals.
        handleOperand(Var);
        // If the operand has already been cloned into JITMod then we
        // need to use the cloned value in the VMap.
        VMap[Arg] = getMappedValue(Var);
      }
    }
  }

  void handleReturnInst(Instruction *I) {
    ResumeAfter = InlinedCalls.back();
    InlinedCalls.pop_back();
    if (RecCallDepth > 0) {
      RecCallDepth -= 1;
      return;
    }
    // Replace the return variable of the call with its return value.
    // Since the return value will have already been copied over to the
    // JITModule, make sure we look up the copy.
    auto OldRetVal = ((ReturnInst *)&*I)->getReturnValue();
    if (OldRetVal != nullptr) {
      assert(ResumeAfter.hasValue());
      VMap[get<1>(ResumeAfter.getValue())] = getMappedValue(OldRetVal);
    }
  }

  void handlePHINode(Instruction *I, Function *F, size_t Idx) {
    assert(Idx > 0);
    auto LBIt = F->begin();
    std::advance(LBIt, BBs[Idx - 1]);
    BasicBlock *LastBlock = &*LBIt;
    Value *V = ((PHINode *)&*I)->getIncomingValueForBlock(LastBlock);
    VMap[&*I] = getMappedValue(V);
  }

  Function *createJITFunc(vector<Value *> *TraceInputs) {
    // Compute a name for the trace.
    uint64_t TraceIdx = getNewTraceIdx();
    TraceName = string(TRACE_FUNC_PREFIX) + to_string(TraceIdx);

    // Create the function.
    std::vector<Type *> InputTypes;
    for (auto Val : *TraceInputs)
      InputTypes.push_back(Val->getType());
    llvm::FunctionType *FType = llvm::FunctionType::get(
        Type::getVoidTy(JITMod->getContext()), InputTypes, false);
    llvm::Function *JITFunc = llvm::Function::Create(
        FType, Function::InternalLinkage, TraceName, JITMod);
    JITFunc->setCallingConv(CallingConv::C);

    return JITFunc;
  }

  // Variables that are used (but not defined) inbetween starting and stopping
  // tracing need to be replaced with function arguments which the user passes
  // into the compiled trace. This loop creates a mapping from those original
  // variables to the function arguments of the compiled trace function.
  void mapTraceInputs(vector<Value *> &TraceInputs, Function *JITFunc) {
    for (size_t Idx = 0; Idx < TraceInputs.size(); Idx++) {
      Value *OldVal = TraceInputs[Idx];
      Value *NewVal = JITFunc->getArg(Idx);
      assert(NewVal->getType()->isPointerTy());
      VMap[OldVal] = NewVal;
    }
  }

public:
  // Store virtual addresses for called functions.
  std::map<StringRef, uint64_t> globalMappings;
  // The function name of this trace.
  string TraceName;
  // Mapping from AOT instructions to JIT instructions.
  ValueToValueMapTy VMap;
#ifndef NDEBUG
  // Reverse mapping for debugging.
  ValueToValueMapTy RevVMap;
#endif

  JITModBuilder(Module *AOTMod, char *FuncNames[], size_t BBs[], size_t Len,
                char *FAddrKeys[], uint64_t FAddrVals[], size_t FAddrLen)
      : Builder(AOTMod->getContext()) {
    this->AOTMod = AOTMod;
    this->FuncNames = FuncNames;
    this->BBs = BBs;
    this->Len = Len;
    this->FAddrKeys = FAddrKeys;
    this->FAddrVals = FAddrVals;
    this->FAddrLen = FAddrLen;

    JITMod = new Module("", AOTMod->getContext());
  }

  // FIXME: this function needs to be refactored.
  // https://github.com/ykjit/yk/issues/385
  Module *createModule() {
    LLVMContext &JITContext = JITMod->getContext();
    // Find the trace inputs.
    auto TraceInputs =
        getTraceInputs(AOTMod->getFunction(FuncNames[0]), BBs[0]);

    // Create function to store compiled trace.
    Function *JITFunc = createJITFunc(&TraceInputs);

    // Add entries to the VMap for variables defined outside of the trace.
    mapTraceInputs(TraceInputs, JITFunc);

    // Create entry block and setup builder.
    auto DstBB = BasicBlock::Create(JITContext, "", JITFunc);
    Builder.SetInsertPoint(DstBB);

    // Iterate over the trace and stitch together all traced blocks.
    for (size_t Idx = 0; Idx < Len; Idx++) {
      auto FuncName = FuncNames[Idx];

      if (ExpectUnmappable && (FuncName == nullptr)) {
        ExpectUnmappable = false;
        continue;
      }
      assert(FuncName != nullptr);

      // Get a traced function so we can extract blocks from it.
      Function *F = AOTMod->getFunction(FuncName);
      if (!F)
        errx(EXIT_FAILURE, "can't find function %s", FuncName);

      // Skip to the correct block.
      auto It = F->begin();
      std::advance(It, BBs[Idx]);
      BasicBlock *BB = &*It;

      // Iterate over all instructions within this block and copy them over
      // to our new module.
      for (size_t CurInstrIdx = 0; CurInstrIdx < BB->size(); CurInstrIdx++) {
        // If we've returned from a call, skip ahead to the instruction where
        // we left off.
        if (ResumeAfter.hasValue() != 0) {
          CurInstrIdx = std::get<0>(ResumeAfter.getValue()) + 1;
          ResumeAfter.reset();
        }
        auto I = BB->begin();
        std::advance(I, CurInstrIdx);
        assert(I != BB->end());

        // Skip calls to debug intrinsics (e.g. @llvm.dbg.value). We don't
        // currently handle debug info and these "pseudo-calls" cause our blocks
        // to be prematurely terminated.
        if (isa<DbgInfoIntrinsic>(I))
          continue;

        if (isa<CallInst>(I)) {
          CallInst *CI = cast<CallInst>(I);
          Function *CF = CI->getCalledFunction();
          if (CF == nullptr) {
            // The target isn't statically known, so we can't inline the callee.
          } else if (CF->getName() == YKTRACE_START) {
            StartTracingInstr = &*CI;
            continue;
          } else if (CF->getName() == YKTRACE_STOP) {
            finalise(&Builder);
            return JITMod;
          } else if (StartTracingInstr != nullptr) {
            handleCallInst(CI, CF, CurInstrIdx);
            break;
          }
        }

        // We don't start copying instructions into the JIT module until we've
        // seen the call to YKTRACE_START.
        if (StartTracingInstr == nullptr)
          continue;

        if ((isa<llvm::BranchInst>(I)) || isa<SwitchInst>(I)) {
          // FIXME Replace all potential CFG divergence with guards.
          continue;
        }

        if (isa<ReturnInst>(I)) {
          handleReturnInst(&*I);
          break;
        }

        if (RecCallDepth > 0) {
          // We are currently ignoring an inlined function.
          continue;
        }

        if (isa<PHINode>(I)) {
          handlePHINode(&*I, F, Idx);
          continue;
        }

        // If execution reaches here, then the instruction I is to be copied
        // into JITMod.
        copyInstruction(&Builder, (Instruction *)&*I);
      }
    }

    // If we fell out of the loop, then we never saw YKTRACE_STOP.
    return NULL;
  }

  void handleOperand(Value *Op) {
    if (VMap.find(Op) == VMap.end()) {
      // The operand is undefined in JITMod.
      Type *OpTy = Op->getType();
      if (isa<llvm::AllocaInst>(Op)) {
        // In the AOT module, the operand is allocated on the stack with
        // an `alloca`, but this variable is as-yet undefined in the JIT
        // module.
        //
        // This happens because LLVM has a tendency to move allocas up to
        // the first block of a function, and if we didn't trace that
        // block (e.g. we started tracing in a later block), then we will
        // have missed those allocations. In these cases we materialise
        // the allocations as we see them used in code that *was* traced.
        Value *Alloca = Builder.CreateAlloca(OpTy->getPointerElementType(),
                                             OpTy->getPointerAddressSpace());
        VMap[Op] = Alloca;
      } else if (isa<ConstantExpr>(Op)) {
        // A `ConstantExpr` may contain operands that require remapping, e.g.
        // global variables. Iterate over all operands and recursively call
        // `handleOperand` on them, then generate a new `ConstantExpr` with
        // the remapped operands.
        ConstantExpr *CExpr = cast<ConstantExpr>(Op);
        std::vector<Constant *> NewCEOps;
        for (unsigned CEOpIdx = 0; CEOpIdx < CExpr->getNumOperands();
             CEOpIdx++) {
          Value *CEOp = CExpr->getOperand(CEOpIdx);
          handleOperand(CEOp);
          NewCEOps.push_back(cast<Constant>(getMappedValue(CEOp)));
        }
        Constant *NewCExpr = CExpr->getWithOperands(NewCEOps);
        VMap[CExpr] = NewCExpr;
      } else if (isa<GlobalVariable>(Op)) {
        // If there's a reference to a GlobalVariable, copy it over to the
        // new module.
        GlobalVariable *OldGV = cast<GlobalVariable>(Op);
        // Global variable is a constant so just copy it into the trace.
        // We don't need to check if this global already exists, since
        // we're skipping any operand that's already been cloned into
        // the VMap.
        GlobalVariable *GV = new GlobalVariable(
            *JITMod, OldGV->getValueType(), OldGV->isConstant(),
            OldGV->getLinkage(), (Constant *)nullptr, OldGV->getName(),
            (GlobalVariable *)nullptr, OldGV->getThreadLocalMode(),
            OldGV->getType()->getAddressSpace());
        VMap[OldGV] = GV;
        if (OldGV->isConstant()) {
          GV->copyAttributesFrom(&*OldGV);
          cloned_globals.push_back(OldGV);
        }
      } else if ((isa<Constant>(Op)) || (isa<InlineAsm>(Op))) {
        // Constants and inline asm don't need to be mapped.
      } else if (Op == StartTracingInstr) {
        // The value generated by StartTracingInstr is the thread tracer.
        // At some optimisation levels, this gets stored in an alloca'd
        // stack space. Since we've stripped the instruction that
        // generates that value (from the JIT module), we have to make a
        // dummy stack slot to keep LLVM happy.
        Value *NullVal = Constant::getNullValue(OpTy);
        VMap[Op] = NullVal;
      } else {
        dumpValueAndExit("don't know how to handle operand", Op);
      }
    }
  }

  void copyInstruction(IRBuilder<> *Builder, Instruction *I) {
    // Before copying an instruction, we have to scan the instruction's
    // operands checking that each is defined in JITMod.
    for (unsigned OpIdx = 0; OpIdx < I->getNumOperands(); OpIdx++) {
      Value *Op = I->getOperand(OpIdx);
      handleOperand(Op);
    }

    // Shortly we will copy the instruction into the JIT module. We start by
    // cloning the instruction.
    auto NewInst = &*I->clone();

    // FIXME: For now we strip debugging meta-data from the JIT module just
    // so that the module will verify and compile. In the long run we should
    // include the debug info for the trace code. This would entail copying
    // over the various module-level debugging declarations that are
    // dependencies of instructions with !dbg meta-data attached.
    if (NewInst->hasMetadata()) {
      SmallVector<std::pair<unsigned, MDNode *>> InstrMD;
      NewInst->getAllMetadata(InstrMD);
      for (auto &MD : InstrMD) {
        if (MD.first != LLVMContext::MD_dbg)
          continue;
        NewInst->setMetadata(MD.first, NULL);
      }
    }

    // Since the instruction operands still reference values from the AOT
    // module, we must remap them to point to new values in the JIT module.
    llvm::RemapInstruction(NewInst, VMap, RF_NoModuleLevelChanges);
    VMap[&*I] = NewInst;

#ifndef NDEBUG
    RevVMap[NewInst] = &*I;
#endif

    // And finally insert the new instruction into the JIT module.
    Builder->Insert(NewInst);
  }

  // Finalise the JITModule by adding a return instruction and initialising
  // global variables.
  void finalise(IRBuilder<> *Builder) {
    Builder->CreateRetVoid();

    // Fix initialisers/referrers for copied global variables.
    // FIXME Do we also need to copy Linkage, MetaData, Comdat?
    for (GlobalVariable *G : cloned_globals) {
      GlobalVariable *NewGV = cast<GlobalVariable>(VMap[G]);
      if (G->isDeclaration())
        continue;

      if (G->hasInitializer())
        NewGV->setInitializer(MapValue(G->getInitializer(), VMap));
    }
  }
};
