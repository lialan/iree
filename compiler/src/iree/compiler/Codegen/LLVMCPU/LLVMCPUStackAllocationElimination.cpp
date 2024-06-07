// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {
/// Selects the lowering strategy for a hal.executable.variant operation.
class LLVMCPUStackAllocationEliminationPass
    : public LLVMCPUStackAllocationEliminationBase<
          LLVMCPUStackAllocationEliminationPass> {
public:
  LLVMCPUStackAllocationEliminationPass() = default;
  LLVMCPUStackAllocationEliminationPass(
      const LLVMCPUStackAllocationEliminationPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<LLVM::LLVMDialect>();
    // clang-format on
  }

  void runOnOperation() override;
};
} // namespace


void LLVMCPUStackAllocationEliminationPass::runOnOperation() {
  return;
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUStackAllocationEliminationPass() {
  return std::make_unique<LLVMCPUStackAllocationEliminationPass>();
}

} // namespace mlir::iree_compiler
