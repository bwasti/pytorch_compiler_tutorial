// All we need to understand PyTorch
#include <torch/csrc/jit/ir.h>
// CompleteArgumentSpec (useful for caching)
#include <torch/csrc/jit/argument_spec.h>
// Our assembler
#include <asmjit/asmjit.h>

using CompiledCode = std::function<std::vector<torch::jit::IValue>(
    at::ArrayRef<torch::jit::IValue>&)>;
class RegisterManager;

class PointwiseCompiler {
 public:
  PointwiseCompiler(const torch::jit::Node* node)
      : subgraph_(node->g(torch::jit::attr::Subgraph)) {}
  void run(torch::jit::Stack& stack);
  static bool supported(const torch::jit::Node* node);

 private:
  CompiledCode emitOperation(
      const torch::jit::Node* node,
      const std::set<const torch::jit::Node*>& seen,
      asmjit::X86Assembler& assembler,
      RegisterManager& reg_manager);
  CompiledCode compile(at::ArrayRef<torch::jit::IValue>&);
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, CompiledCode> cache_;
  asmjit::JitRuntime jit_runtime_;
};
