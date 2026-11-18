"""Graph rewrites applied after export to work around runtime gaps.

rewrite_deconv_as_depth_to_space
    TensorRT 11.4 has no bfloat16 implementation of ConvTranspose, and folds
    fp32 casts placed around it back into a bf16 deconvolution. A ConvTranspose
    whose kernel equals its stride (no padding, no dilation, group 1) is exactly
    a 1x1 Conv to C_out*k*k channels followed by DepthToSpace(blocksize=k,
    mode="CRD"), both of which TensorRT supports in bf16:

        out[co, h*k+i, w*k+j] = sum_ci x[ci, h, w] * W[ci, co, i, j] + b[co]

    The rewrite is exact (same arithmetic, same summation per output element);
    it is verified numerically on the fp32 graph, see README.

cast_constants_to_input_dtype
    In a 16-bit refiner graph the exporter's optimiser folds some Conv +
    BatchNorm pairs into a Conv whose folded weights are fp32 initializers
    while its input is bf16/fp16, and leaves BatchNormalization parameters
    fp32 next to a 16-bit input. ONNX requires one type per node; TensorRT
    rejects the mismatch. A Cast to the input's type is inserted on each such
    constant (weights stay fp32 on disk, as in the rest of a 16-bit graph).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper


def rewrite_deconv_as_depth_to_space(onnx_path: Path) -> int:
    """Rewrite eligible ConvTranspose nodes in place. Returns how many were rewritten."""
    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path), load_external_data=True)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}

    def constant(name: str) -> tuple[np.ndarray, int]:
        """fp32 value of a constant input and the dtype the graph uses it in."""
        if name in inits:
            t = inits[name]
            return numpy_helper.to_array(t).astype(np.float32), t.data_type
        cast = producer[name]
        assert cast.op_type == "Cast" and cast.input[0] in inits, f"unexpected constant source for {name}"
        to = next(a.i for a in cast.attribute if a.name == "to")
        return numpy_helper.to_array(inits[cast.input[0]]).astype(np.float32), to

    new_nodes, new_inits, rewritten = [], [], 0
    for node in graph.node:
        if node.op_type != "ConvTranspose":
            new_nodes.append(node)
            continue
        attr = {a.name: (list(a.ints) if a.ints else a.i) for a in node.attribute}
        k = attr["kernel_shape"]
        eligible = (
            k[0] == k[1]
            and attr.get("strides") == k
            and not any(attr.get("pads", [0]))
            and not any(attr.get("output_padding", [0]))
            and all(d == 1 for d in attr.get("dilations", [1]))
            and attr.get("group", 1) == 1
            and len(node.input) == 3
        )
        if not eligible:
            new_nodes.append(node)
            continue
        k = k[0]
        W, w_dtype = constant(node.input[1])  # [C_in, C_out, k, k]
        b, _ = constant(node.input[2])  # [C_out]
        c_in, c_out = W.shape[0], W.shape[1]
        # Conv weight row (co*k*k + i*k + j) holds W[:, co, i, j]: DepthToSpace "CRD" order.
        W2 = np.ascontiguousarray(W.transpose(1, 2, 3, 0).reshape(c_out * k * k, c_in, 1, 1))
        b2 = np.repeat(b, k * k)
        base = node.name or node.output[0]
        w_name, b_name = f"{base}__d2s_weight", f"{base}__d2s_bias"
        new_inits += [numpy_helper.from_array(W2, w_name), numpy_helper.from_array(b2, b_name)]
        if w_dtype != TP.FLOAT:
            # Keep weights fp32 on disk and cast, like the rest of a 16-bit graph.
            new_nodes.append(helper.make_node("Cast", [w_name], [w_name + "_cast"], to=w_dtype))
            new_nodes.append(helper.make_node("Cast", [b_name], [b_name + "_cast"], to=w_dtype))
            w_name, b_name = w_name + "_cast", b_name + "_cast"
        mid = f"{base}__d2s_conv"
        new_nodes.append(
            helper.make_node("Conv", [node.input[0], w_name, b_name], [mid], name=f"{base}__as_conv", kernel_shape=[1, 1])
        )
        new_nodes.append(
            helper.make_node(
                "DepthToSpace", [mid], list(node.output), name=f"{base}__as_d2s", blocksize=k, mode="CRD"
            )
        )
        rewritten += 1

    if not rewritten:
        return 0
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    # Drop constants (and their casts) that only fed the removed nodes.
    while True:
        used = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
        dead = [n for n in graph.node if n.op_type == "Cast" and not any(o in used for o in n.output)]
        if not dead:
            break
        for n in dead:
            graph.node.remove(n)
    used = {i for n in graph.node for i in n.input}
    for name in [i.name for i in graph.initializer if i.name not in used]:
        graph.initializer.remove(next(i for i in graph.initializer if i.name == name))

    data_name = onnx_path.name + ".data"
    # onnx appends to an existing external-data file; start clean.
    (onnx_path.parent / data_name).unlink(missing_ok=True)
    onnx.save(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    return rewritten


def cast_constants_to_input_dtype(onnx_path: Path, op_types=("Conv", "BatchNormalization")) -> int:
    """Cast fp32 initializers feeding 16-bit Conv/BatchNormalization nodes to the
    node's input type, in place. Returns how many constants were cast."""
    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path), load_external_data=True)
    graph = model.graph
    init_dtype = {i.name: i.data_type for i in graph.initializer}
    dtype = {v.name: v.type.tensor_type.elem_type for v in list(graph.value_info) + list(graph.input)}
    for n in graph.node:  # Cast outputs are not always in value_info
        if n.op_type == "Cast":
            dtype[n.output[0]] = next(a.i for a in n.attribute if a.name == "to")

    new_nodes, casts = [], {}
    for node in graph.node:
        if node.op_type in op_types:
            want = dtype.get(node.input[0], TP.FLOAT)
            if want in (TP.BFLOAT16, TP.FLOAT16):
                for idx in range(1, len(node.input)):
                    name = node.input[idx]
                    if init_dtype.get(name) == TP.FLOAT:
                        key = (name, want)
                        if key not in casts:
                            casts[key] = f"{name}__as_{TP.DataType.Name(want).lower()}"
                            new_nodes.append(helper.make_node("Cast", [name], [casts[key]], to=want))
                        node.input[idx] = casts[key]
        new_nodes.append(node)
    if not casts:
        return 0
    del graph.node[:]
    graph.node.extend(new_nodes)
    data_name = onnx_path.name + ".data"
    # onnx appends to an existing external-data file; start clean.
    (onnx_path.parent / data_name).unlink(missing_ok=True)
    onnx.save(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    return len(casts)
