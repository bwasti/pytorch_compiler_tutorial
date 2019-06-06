This is the codebase associated with the [PyTorch JIT compiler tutorial](https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch).

## Build and Test

A recent version of PyTorch will likely need to have been installed (either nightly or built from source).

```
git clone https://github.com/bwasti/pytorch_compiler_tutorial.git --recursive
cd pytorch_compiler_tutorial
./build.sh
PYTHONPATH=build python test.py
```
