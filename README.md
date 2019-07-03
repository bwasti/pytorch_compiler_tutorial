This is the codebase associated with the [PyTorch JIT compiler tutorial](https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch).

## Build and Test

A recent version of PyTorch will likely need to have been installed (either nightly or built from source).

```
git clone https://github.com/bwasti/pytorch_compiler_tutorial.git --recursive
cd pytorch_compiler_tutorial
./build.sh
PYTHONPATH=build python test.py
```

Expect to see this output:

```
-- Default IR --
 graph(%a.1 : Float(*),
      %b.1 : Float(*)):
  %c.1 : Float(*) = aten::mul(%a.1, %b.1) # test.py:20:7
  %a.3 : Float(*) = aten::mul(%c.1, %c.1) # test.py:21:7
  %a.5 : Float(*) = aten::mul(%c.1, %a.3) # test.py:22:7
  return (%a.5)

Default version took 26.74ms

-- Transformed IR --
 graph(%a.1 : Float(*),
      %b.1 : Float(*)):
  %a.5 : Float(*) = pw::CompilationGroup_0(%a.1, %b.1)
  return (%a.5)
with pw::CompilationGroup_0 = graph(%4 : Float(*),
      %5 : Float(*)):
  %c.1 : Float(*) = aten::mul(%4, %5) # test.py:34:7
  %a.3 : Float(*) = aten::mul(%c.1, %c.1) # test.py:35:7
  %a.5 : Float(*) = aten::mul(%c.1, %a.3) # test.py:36:7
  return (%a.5)

Compiled version took 8.20ms
```
