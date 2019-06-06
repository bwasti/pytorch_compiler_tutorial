#include <pybind11/pybind11.h>

// Register our compiler as handling a
// specific type of operator
#include <torch/csrc/jit/custom_operator.h>

// Register a pass to convert the IR into one with our operator
#include <torch/csrc/jit/pass_manager.h>
// CustomFuseGraph is a helper to use simple whitelisting
#include <torch/csrc/jit/passes/graph_fuser.h>

#include "compiler.h"

namespace py = pybind11;
using namespace torch::jit;

PYBIND11_MODULE(pointwise_compiler, m) {
  // PyTorch makes heavy use of interned strings, which are called Symbols
  const auto pointwise_compiler_symbol =
      Symbol::fromQualString("pw::CompilationGroup");

  // Let's hook up the compiler!

  // First, register a pass that will coalesce operators we can handle
  // into a single operator containing a subgraph.
  RegisterPass pass([pointwise_compiler_symbol](std::shared_ptr<Graph>& g) {
    CustomFuseGraph(g, PointwiseCompiler::supported, pointwise_compiler_symbol);
  });

  // We are only dealing with pure operations (no aliasing or in place
  // mutation), so our subgraph will always be pure.
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(AliasAnalysisKind::PURE);

  RegisterOperators op({Operator(
      pointwise_compiler_symbol,
      [](const Node* node) {
        auto compiler = std::make_shared<PointwiseCompiler>(node);
        return [compiler](Stack& stack) {
          compiler->run(stack);
          return 0;
        };
      },
      options)});
}
