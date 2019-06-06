import torch
import time

def benchmark(f):
  A_ = torch.randn(1024)
  B_ = torch.randn(1024)
  # Warmup
  for _ in range(10):
    _ = f(A_,B_)
  t = time.time()
  for _ in range(100):
    _ = f(A_,B_)
  return time.time() - t

A = torch.randn(1024)
B = torch.randn(1024)

@torch.jit.script
def foo_jit(a, b):
  c = a.mul(b)
  a = c.mul(c)
  a = c.mul(a)
  return a

print("-- Default IR --\n", foo_jit.graph_for(A,B))
C_jit = foo_jit(A,B)
print("Default version took {:.2f}ms".format(1000 * benchmark(foo_jit)))

import pointwise_compiler
print()

@torch.jit.script
def foo_compiled(a, b):
  c = a.mul(b)
  a = c.mul(c)
  a = c.mul(a)
  return a

print("-- Transformed IR --\n", foo_compiled.graph_for(A,B))
C_compiled = foo_compiled(A,B)
print("Compiled version took {:.2f}ms".format(1000 * benchmark(foo_compiled)))

assert torch.allclose(C_jit, C_compiled)
