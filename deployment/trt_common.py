"""TensorRT helpers: a collecting logger and a minimal engine runner.

Kept separate from common.py so the export scripts do not need TensorRT.
Written against TensorRT 11 (always strongly typed; precision comes from the
ONNX graph, there are no FP16/INT8 builder flags or calibrators).
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT8: torch.int8,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.BOOL: torch.bool,
}


class CollectingLogger(trt.ILogger):
    """Prints warnings and above, and keeps them for the build log."""

    def __init__(self, verbose: bool = False):
        trt.ILogger.__init__(self)
        self.verbose = verbose
        self.messages: list[dict] = []

    def log(self, severity, msg):
        if severity <= trt.ILogger.Severity.WARNING:
            self.messages.append({"severity": severity.name, "message": msg})
            print(f"[TRT {severity.name}] {msg}", flush=True)
        elif self.verbose and severity <= trt.ILogger.Severity.INFO:
            print(f"[TRT {severity.name}] {msg}", flush=True)


class TrtRunner:
    """Run a serialized engine with torch CUDA tensors as I/O buffers."""

    def __init__(self, engine_path: str | Path):
        self.logger = CollectingLogger()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        assert self.engine is not None, f"failed to load {engine_path}"
        self.context = self.engine.create_execution_context()
        # A dedicated stream: on the default stream TensorRT adds extra
        # cudaStreamSynchronize calls (and warns about it).
        self.stream = torch.cuda.Stream()
        self.input_names, self.output_names = [], []
        self.dtypes, self.shapes = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.dtypes[name] = _TORCH_DTYPE[self.engine.get_tensor_dtype(name)]
            self.shapes[name] = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self.outputs = {
            n: torch.empty(self.shapes[n], dtype=self.dtypes[n], device="cuda")
            for n in self.output_names
        }
        for n, t in self.outputs.items():
            self.context.set_tensor_address(n, t.data_ptr())

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Returns views of the runner's output buffers (overwritten on the next call)."""
        held = []
        for n in self.input_names:
            t = inputs[n].to(device="cuda", dtype=self.dtypes[n]).contiguous()
            assert tuple(t.shape) == self.shapes[n], (n, t.shape, self.shapes[n])
            held.append(t)
            self.context.set_tensor_address(n, t.data_ptr())
        torch.cuda.current_stream().synchronize()  # inputs are ready
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        assert ok, "TensorRT execution failed"
        self.stream.synchronize()
        return self.outputs
