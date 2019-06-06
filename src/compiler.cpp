#include "compiler.h"

#include <stack>

using namespace torch::jit;

// This class will be used to allocate registers
// as we traverse the frontier of the PyTorch IR graph.
class RegisterManager {
 public:
  RegisterManager() = default;
  asmjit::X86Gp getFreeAddrReg() {
    TORCH_CHECK(free_addr_regs.size() > 0);
    auto reg = free_addr_regs.back();
    free_addr_regs.pop_back();
    return reg;
  }

  asmjit::X86Xmm getFreeValueReg() {
    TORCH_CHECK(free_value_regs.size() > 0);
    auto reg = free_value_regs.back();
    free_value_regs.pop_back();
    return reg;
  }

  asmjit::X86Gp getAddrReg(const Value* v) {
    TORCH_CHECK(addr_regs.find(v) != addr_regs.end());
    return addr_regs[v];
  }

  asmjit::X86Xmm getValueReg(const Value* v) {
    TORCH_CHECK(value_regs.find(v) != value_regs.end());
    return value_regs[v];
  }

  void mapReg(const Value* v, asmjit::X86Gp gp) {
    addr_regs[v] = gp;
  }

  void mapReg(const Value* v, asmjit::X86Xmm xmm) {
    value_regs[v] = xmm;
  }

  void free(asmjit::X86Gp reg) {
    free_addr_regs.push_back(reg);
  }

  void free(asmjit::X86Xmm reg) {
    free_value_regs.push_back(reg);
  }

 private:
  std::unordered_map<const Value*, asmjit::X86Gp> addr_regs;
  std::unordered_map<const Value*, asmjit::X86Xmm> value_regs;

  std::vector<asmjit::X86Gp> free_addr_regs = {
      asmjit::x86::rsi,
      asmjit::x86::rdx,
      asmjit::x86::r8,
      asmjit::x86::r9,
      asmjit::x86::r10,
      asmjit::x86::r11,
  };

  std::vector<asmjit::X86Xmm> free_value_regs = {
      asmjit::x86::xmm0,
      asmjit::x86::xmm1,
      asmjit::x86::xmm2,
      asmjit::x86::xmm3,
      asmjit::x86::xmm4,
      asmjit::x86::xmm5,
      asmjit::x86::xmm6,
      asmjit::x86::xmm7,
  };
};

bool PointwiseCompiler::supported(const torch::jit::Node* node) {
  switch (node->kind()) {
    case aten::mul:
      return true;
    default:
      return false;
  }
  return false;
}

void PointwiseCompiler::run(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<IValue> inputs = last(stack, num_inputs);

  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};
  if (cache_.find(spec) == cache_.end()) {
    cache_[spec] = compile(inputs);
  }

  // Run the compiled function!
  auto outputs = cache_[spec](inputs);

  drop(stack, num_inputs);
  for (auto& output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(IValue(var));
  }
}

CompiledCode PointwiseCompiler::emitOperation(
    const Node* node,
    const std::set<const Node*>& seen,
    asmjit::X86Assembler& assembler,
    RegisterManager& reg_manager) {
  switch (node->kind()) {
    case aten::mul: {
      auto A = node->inputs()[0];
      auto C_reg = reg_manager.getValueReg(A);
      for (auto use : A->uses()) {
        if (seen.find(use.user) != seen.end()) {
          C_reg = reg_manager.getFreeValueReg();
          assembler.movaps(C_reg, reg_manager.getValueReg(A));
        }
      }
      auto B = node->inputs()[1];
      assembler.mulss(C_reg, reg_manager.getValueReg(B));
      reg_manager.mapReg(node->outputs()[0], C_reg);
    }
  }
  for (auto& input : node->inputs()) {
    bool used = true;
    for (auto use : input->uses()) {
      if (seen.find(use.user) == seen.end()) {
        used = false;
      }
    }
    if (used) {
      reg_manager.free(reg_manager.getValueReg(input));
    }
  }
}

CompiledCode PointwiseCompiler::compile(
    at::ArrayRef<torch::jit::IValue>& inputs) {
  // First we run through some checks to make sure the inputs are Tensors and
  // that the implied semantics are pointwise.
  TORCH_CHECK(inputs.size(), "Need at least one input.");
  for (const auto& input : inputs) {
    TORCH_CHECK(input.isTensor(), "Compiler can only handle Tensor inputs.");
  }
  auto size = inputs[0].toTensor().numel();
  for (const auto& input : inputs) {
    TORCH_CHECK(
        input.toTensor().numel() == size,
        "Compiler can only handle pointwise operations without broadcasting.");
  }

  // Then we setup code generation utilities.
  auto reg_manager = RegisterManager();
  asmjit::CodeHolder code;
  code.init(jit_runtime_.getCodeInfo());
  asmjit::StringLogger asm_logger;
  code.setLogger(&asm_logger);
  asmjit::X86Assembler assembler(&code);

  const bool isWinOS = static_cast<bool>(ASMJIT_OS_WINDOWS);
  asmjit::X86Gp pointers = isWinOS ? asmjit::x86::rcx : asmjit::x86::rdi;

  // Move all the input Tensor addresses into registers
  for (auto i = 0; i < inputs.size(); ++i) {
    auto reg = reg_manager.getFreeAddrReg();
    auto mem_ptr = asmjit::x86::ptr(pointers, i * sizeof(void*));
    reg_manager.mapReg(subgraph_->inputs()[i], reg);
    assembler.mov(reg, mem_ptr);
  }

  // Do the same with output Tensors
  for (auto i = 0; i < subgraph_->outputs().size(); ++i) {
    auto reg = reg_manager.getFreeAddrReg();
    auto mem_ptr =
        asmjit::x86::ptr(pointers, (i + inputs.size()) * sizeof(void*));
    reg_manager.mapReg(subgraph_->outputs()[i], reg);
    assembler.mov(reg, mem_ptr);
  }

  // Setup a label for looping
  auto iter = reg_manager.getFreeAddrReg();
  assembler.mov(iter, 0);
  auto loop_label = assembler.newLabel();
  assembler.bind(loop_label);

  // Now we iterate through the nodes, keeping track of which ones we've
  // seen.  If the input to a node has been totally consumed (no nodes
  // we haven't seen will use it), we free that register.
  std::set<const Node*> seen;

  for (auto input : subgraph_->inputs()) {
    auto reg = reg_manager.getFreeValueReg();
    assembler.movd(
        reg, asmjit::x86::ptr(reg_manager.getAddrReg(input), iter, 2));
    reg_manager.mapReg(input, reg);
  }

  // Iterating over graph nodes is guaranteed to be topologically sorted
  for (auto node : subgraph_->nodes()) {
    seen.insert(node);
    emitOperation(node, seen, assembler, reg_manager);
  }

  // Store all the output values into memory.
  for (auto output : subgraph_->outputs()) {
    assembler.movd(
        asmjit::x86::ptr(reg_manager.getAddrReg(output), iter, 2),
        reg_manager.getValueReg(output));
  }

  assembler.add(iter, 1);
  assembler.cmp(iter, size);
  assembler.jb(loop_label);

  assembler.ret();

  // Now we bind a function the assembly we generated.
  void (*fn)(void**);
  asmjit::Error err = jit_runtime_.add(&fn, &code);
  TORCH_CHECK(
      !err,
      "Couldn't create function, asm:\n",
      std::string(asm_logger.getString()));

  // This function wraps the function pointer we bound our assembly to
  // Adheres to the CompiledCode interface defined in compiler.h
  auto compiled_func = [this, fn, size](at::ArrayRef<IValue>& inputs) {
    std::vector<void*> args;
    for (auto input : inputs) {
      TORCH_CHECK(input.isTensor());
      TORCH_CHECK(input.toTensor().is_contiguous());
      TORCH_CHECK(input.toTensor().device().is_cpu());
      args.emplace_back(input.toTensor().data_ptr());
    }
    std::vector<IValue> outputs;
    for (auto output : subgraph_->outputs()) {
      outputs.emplace_back(at::empty({size}));
    }
    for (auto output : outputs) {
      args.emplace_back(output.toTensor().data_ptr());
    }

    // Run the function
    fn(args.data());

    return outputs;
  };

  return compiled_func;
}
