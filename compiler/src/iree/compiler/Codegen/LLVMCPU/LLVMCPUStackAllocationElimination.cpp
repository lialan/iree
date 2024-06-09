// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-llvmcpu-stack-elimination"

static llvm::cl::opt<int> clMaxAllocationSizeInBytes(
    "iree-llvmcpu-memory-allocation-limit",
    llvm::cl::desc("maximum allowed memory allocation size in bytes"),
    llvm::cl::init(65536));

namespace mlir::iree_compiler {

namespace {
class LLVMCPUStackAllocationEliminationPass
    : public PassWrapper<LLVMCPUStackAllocationEliminationPass,
                         OperationPass<ModuleOp>> {
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
  
  /// Collect allocas in this function into `allocas`. Does not allow dynamic allocation.
  void collectAllocas(ModuleOp &);
  size_t getSingleAllocaSize(memref::AllocaOp&) const;

  LogicalResult checkTotalAllocaSize() const;
  void mapAllocasToMemory(mlir::FunctionOpInterface funcOp);
  size_t getAllocationSize(memref::AllocaOp &allocaOp) const;

  SmallVector<memref::AllocaOp> allocas;

};
} // namespace

size_t getAllocationSize(memref::AllocaOp& allocaOp) {
  size_t allocaSize = 1;
  auto allocaType = llvm::cast<ShapedType>(allocaOp.getType());
  for (auto dimSize : allocaType.getShape()) {
    assert(!ShapedType::isDynamic(dimSize) &&
           "Dynamic allocation is not allowed");
    allocaSize *= dimSize;
  }
  return allocaSize;
}

LogicalResult LLVMCPUStackAllocationEliminationPass::checkTotalAllocaSize() const {
  // sum total sizes in collected allocas
  size_t totalAllocaSize =
      std::accumulate(allocas.begin(), allocas.end(), 0,
                      [this](size_t sum, memref::AllocaOp &rhs) {
                        sum += getSingleAllocaSize(rhs);
                      });
  if (totalAllocaSize > clMaxAllocationSizeInBytes) {
    return funcOp.emitOpError("exceeded stack allocation limit of ")
           << clMaxAllocationSizeInBytes.getValue()
           << " bytes for function. Got " << totalAllocaSize << " bytes";
  }
}

void LLVMCPUStackAllocationEliminationPass::runOnOperation() {
  auto moduleOp = getOperation();

  allocas.clear();
  collectAllocas(moduleOp);
  if (failed(checkTotalAllocaSize()))
    return signalPassFailure();

  mapAllocasToMemory();

  return;
}

void LLVMCPUStackAllocationEliminationPass::collectAllocas(ModuleOp& moduleOp) {
  SmallVector<memref::AllocaOp> allocaOps;
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    funcOp.walk(
        [&](memref::AllocaOp allocaOp) { allocaOps.push_back(allocaOp); });
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUStackAllocationEliminationPass() {
  return std::make_unique<LLVMCPUStackAllocationEliminationPass>();
}

} // namespace mlir::iree_compiler
