from setuptools import setup
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension
import pybind11

from distutils.sysconfig import get_python_inc


class BuildExtCUDA(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        default_compiler_so = self.compiler._compile

        def cuda_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                # Use nvcc for .cu files
                self.spawn(
                    [
                        "nvcc",
                        "-c",
                        src,
                        "-o",
                        obj,
                        "-std=c++17",
                        "-O2",
                        "--compiler-options",
                        "'-fPIC'",
                        "-I",
                        pybind11.get_include(),
                        "-I",
                        get_python_inc()
                    ]
                    + [arg for arg in extra_postargs if not (arg.startswith("-g0") or "fvisibility" in arg)]
                )
            else:
                default_compiler_so(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = cuda_compile

        # Remove unsupported flags
        for ext in self.extensions:
            ext.extra_compile_args = [
                flag for flag in ext.extra_compile_args if not (flag.startswith("-g0") or "fvisibility" in flag)
            ]

        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        "gmm_module",
        sources=["gmm_wrapper.cu", "kernels.cu", "csv_read.cpp"],  # CUDA and C++ files
        include_dirs=[pybind11.get_include(), "/usr/local/cuda/include"],  # Add pybind11 and CUDA includes
        library_dirs=["/usr/local/cuda/lib64"],  # CUDA library path
        libraries=["cudart"],  # Link against CUDA runtime
        extra_compile_args=["-O2"],  # Host compiler optimization flags
    ),
]

setup(
    name="gmm_module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtCUDA},
)
