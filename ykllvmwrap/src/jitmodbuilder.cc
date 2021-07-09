// Classes and functions for constructing a new LLVM module from a trace.

using namespace llvm;
using namespace std;

// An atomic counter used to issue compiled traces with unique names.
atomic<uint64_t> NextTraceIdx(0);

#define TRACE_FUNC_PREFIX "__yk_compiled_trace_"
#define YKTRACE_START "__yktrace_start_tracing"
#define YKTRACE_STOP "__yktrace_stop_tracing"

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
      if (CI->getCalledFunction()->getName() == YKTRACE_START) {
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
  // The new module that is being build.
  Module *JITMod;
  // A pointer to the call to YKTRACE_START in the AOT module (once
  // encountered). When this changes from NULL to non-NULL, then we start
  // copying instructions from the AOT module into the JIT module.
  Instruction *StartTracingInstr = nullptr;

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

  Value *getMappedValue(Value *V) {
    if (isa<Constant>(V)) {
      return V;
    } else {
      auto NV = VMap[V];
      return NV;
    }
  }

  Module *createModule(char *FuncNames[], size_t BBs[], size_t Len,
                       Module *AOTMod, char *FAddrKeys[], uint64_t FAddrVals[],
                       size_t FAddrLen) {
    LLVMContext &JITContext = AOTMod->getContext();
    JITMod = new Module("", JITContext);
    uint64_t TraceIdx = NextTraceIdx.fetch_add(1);
    if (TraceIdx == numeric_limits<uint64_t>::max())
      errx(EXIT_FAILURE, "trace index counter overflowed");

    // Get var args from start_tracing call.
    auto Inputs = getTraceInputs(AOTMod->getFunction(FuncNames[0]), BBs[0]);

    std::vector<Type *> InputTypes;
    for (auto Val : Inputs) {
      InputTypes.push_back(Val->getType());
    }

    // Create function to store compiled trace.
    TraceName = string(TRACE_FUNC_PREFIX) + to_string(TraceIdx);
    llvm::FunctionType *FType =
        llvm::FunctionType::get(Type::getVoidTy(JITContext), InputTypes, false);
    llvm::Function *JITFunc = llvm::Function::Create(
        FType, Function::InternalLinkage, TraceName, JITMod);
    JITFunc->setCallingConv(CallingConv::C);

    // Create entry block and setup builder.
    auto DstBB = BasicBlock::Create(JITContext, "", JITFunc);
    llvm::IRBuilder<> Builder(JITContext);
    Builder.SetInsertPoint(DstBB);

    // Variables that are used (but not defined) inbetween start and stop
    // tracing need to be replaced with function arguments which the user passes
    // into the compiled trace. This loop creates a mapping from those original
    // variables to the function arguments of the compiled trace function.
    for (size_t Idx = 0; Idx != Inputs.size(); Idx++) {
      Value *OldVal = Inputs[Idx];
      Value *NewVal = JITFunc->getArg(Idx);
      assert(NewVal->getType()->isPointerTy());
      VMap[OldVal] = NewVal;
    }

    std::vector<CallInst *> inlined_calls;
    CallInst *last_call = nullptr;
    size_t call_stack = 0;
    CallInst *noinline_func = nullptr;

    // Iterate over the trace and stitch together all traced blocks.
    for (size_t Idx = 0; Idx < Len; Idx++) {
      auto FuncName = FuncNames[Idx];

      // FIXME: Deal with holes in the trace, e.g. calls to libc.
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
      for (auto I = BB->begin(); I != BB->end(); I++) {
        // Skip calls to debug intrinsics (e.g. @llvm.dbg.value). We don't
        // currently handle debug info and these "pseudo-calls" cause our blocks
        // to be prematurely terminated.
        if (isa<DbgInfoIntrinsic>(I))
          continue;

        // If we've returned from a call, skip ahead to the instruction where
        // we left off.
        if (last_call != nullptr) {
          if (&*I == last_call) {
            last_call = nullptr;
          }
          continue;
        }

        if (isa<CallInst>(I)) {
          CallInst *CI = cast<CallInst>(&*I);
          Function *CF = CI->getCalledFunction();

          if (CF == nullptr) {
            // It's an external function or an intrinsic. We can't inline it,
            // so we have no option but to copy the call as-is.
          } else if (CF->getName() == YKTRACE_START) {
            StartTracingInstr = &*I;
            continue;
          } else if (CF->getName() == YKTRACE_STOP) {
            finalise(&Builder);
            return JITMod;
          } else {
            StringRef CFName = CF->getName();
            if (AOTMod->getFunction(CFName) != nullptr && call_stack > 0) {
              // When ignoring an inlined function, we need to count other
              // inlined function calls so we know when we left the initial
              // function call.
              call_stack += 1;
              inlined_calls.push_back(CI);
              break;
            }
            // If this is a recursive call that has been inlined, remove the
            // inlined code and turn it into a normal call.
            for (CallInst *cinst : inlined_calls) {
              // Have we inlined this call already? Then this is recursion.
              if (cinst->getCalledFunction() == CF) {
                if (VMap[CF] == nullptr) {
                  // Declare function.
                  auto DeclFunc = llvm::Function::Create(
                      CF->getFunctionType(), GlobalValue::ExternalLinkage,
                      CFName, JITMod);
                  VMap[CF] = DeclFunc;
                  for (size_t i = 0; i < FAddrLen; i++) {
                    char *FName = FAddrKeys[i];
                    uint64_t FAddr = FAddrVals[i];
                    if (strcmp(FName, CFName.data()) == 0) {
                      globalMappings.insert(
                          pair<StringRef, uint64_t>(CFName, FAddr));
                      break;
                    }
                  }
                }
                copyInstruction(&Builder, (Instruction *)&*I);
                noinline_func = CI;
                call_stack = 1;
                break;
              }
            }
            // Skip remainder of this block and remember where we stopped so we
            // can continue from this position after returning from the inlined
            // call.
            if (StartTracingInstr != nullptr) {
              inlined_calls.push_back(CI);
              // During inlining, remap function arguments to the variables
              // passed in by the caller.
              if (call_stack == 0) {
                for (unsigned int i = 0; i < CI->arg_size(); i++) {
                  Value *Var = CI->getArgOperand(i);
                  Value *Arg = CF->getArg(i);
                  // If the operand has already been cloned into JITMod then we
                  // need to use the cloned value in the VMap.
                  if (VMap[Var] != nullptr)
                    Var = VMap[Var];
                  VMap[Arg] = Var;
                }
              }
              break;
            }
          }
        }

        // We don't start copying instructions into the JIT module until we've
        // seen the call to YKTRACE_START.
        if (StartTracingInstr == nullptr)
          continue;

        if (llvm::isa<llvm::BranchInst>(I)) {
          // FIXME Replace all branch instruction with guards.
          continue;
        }

        if (isa<ReturnInst>(I)) {
          last_call = inlined_calls.back();
          inlined_calls.pop_back();
          if (call_stack > 0) {
            call_stack -= 1;
            if (call_stack == 0) {
              last_call = noinline_func;
            }
            continue;
          }
          // Replace the return variable of the call with its return value.
          // Since the return value will have already been copied over to the
          // JITModule, make sure we look up the copy.
          auto OldRetVal = ((ReturnInst *)&*I)->getReturnValue();
          VMap[last_call] = getMappedValue(OldRetVal);
          break;
        }

        if (call_stack > 0) {
          // We are currently ignoring an inlined function.
          continue;
        }

        if (isa<PHINode>(I)) {
          assert(Idx > 0);
          auto LBIt = F->begin();
          std::advance(LBIt, BBs[Idx - 1]);
          BasicBlock *LastBlock = &*LBIt;
          Value *V = ((PHINode *)&*I)->getIncomingValueForBlock(LastBlock);
          VMap[&*I] = getMappedValue(V);
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

  void copyInstruction(IRBuilder<> *Builder, Instruction *I) {
    // Before copying an instruction, we have to scan the instruction's
    // operands checking that each is defined in JITMod.
    for (unsigned OpIdx = 0; OpIdx < I->getNumOperands(); OpIdx++) {
      Value *Op = I->getOperand(OpIdx);
      if (VMap[Op] == nullptr) {
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
          Value *Alloca = Builder->CreateAlloca(OpTy->getPointerElementType(),
                                                OpTy->getPointerAddressSpace());
          VMap[Op] = Alloca;
        } else if (isa<GlobalVariable>(Op)) {
          // If there's a reference to a GlobalVariable, copy it over to the
          // new module.
          GlobalVariable *OldGV = cast<GlobalVariable>(Op);
          if (OldGV->isConstant()) {
            // Global variable is a constant so just copy it into the trace.
            // We don't need to check if this global already exists, since
            // we're skipping any operand that's already been cloned into
            // the VMap.
            GlobalVariable *GV = new GlobalVariable(
                *JITMod, OldGV->getValueType(), OldGV->isConstant(),
                OldGV->getLinkage(), (Constant *)nullptr, OldGV->getName(),
                (GlobalVariable *)nullptr, OldGV->getThreadLocalMode(),
                OldGV->getType()->getAddressSpace());
            GV->copyAttributesFrom(&*OldGV);
            cloned_globals.push_back(OldGV);
            VMap[OldGV] = GV;
          } else {
            // FIXME Allow trace to write to mutable global variables.
            errx(EXIT_FAILURE, "Non-const global variable %s",
                 OldGV->getName().data());
          }
        } else if ((isa<Constant>(Op)) || (isa<InlineAsm>(Op))) {
          // Constants and inline asm can be ID-mapped.
          VMap[Op] = Op;
          continue;
        } else if (Op == StartTracingInstr) {
          // The value generated by StartTracingInstr is the thread tracer.
          // At some optimisation levels, this gets stored in an alloca'd
          // stack space. Since we've stripped the instruction that
          // generates that value (from the JIT module), we have to make a
          // dummy stack slot to keep LLVM happy.
          Value *NullVal = Constant::getNullValue(OpTy);
          VMap[Op] = NullVal;
        } else {
          errx(EXIT_FAILURE, "don't know how to handle operand");
        }
      }
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
