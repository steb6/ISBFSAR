import pycuda.autoinit  # IMPORTANT! LEAVE THIS IMPORT HERE
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Runner:
    def __init__(self, engine):
        G_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(G_LOGGER, '')
        runtime = trt.Runtime(G_LOGGER)

        with open(engine, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)

        # prepare buffer
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 256 x 256 x 3 ( x 1 )
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)  # (256 x 256 x 3 ) x (32 / 4)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        # store
        self.stream = cuda.Stream()
        self.context = engine.create_execution_context()
        self.engine = engine

        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings

    def __call__(self, pc):

        if not isinstance(pc, list):
            pc = [pc]
        for elem, inp in zip(pc, self.inputs):

            elem = elem.ravel()
            np.copyto(inp.host, elem)

        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()

        res = [out.host for out in self.outputs]

        return res
