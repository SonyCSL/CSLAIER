#!/usr/bin/env python2
import ctypes
import platform
from logging import getLogger


logger = getLogger(__name__)


class c_cudaDeviceProp(ctypes.Structure):
    """
    Passed to cudart.cudaGetDeviceProperties()
    """
    _fields_ = [
        ('name', ctypes.c_char * 256),
        ('totalGlobalMem', ctypes.c_size_t),
        ('sharedMemPerBlock', ctypes.c_size_t),
        ('regsPerBlock', ctypes.c_int),
        ('warpSize', ctypes.c_int),
        ('memPitch', ctypes.c_size_t),
        ('maxThreadsPerBlock', ctypes.c_int),
        ('maxThreadsDim', ctypes.c_int * 3),
        ('maxGridSize', ctypes.c_int * 3),
        ('clockRate', ctypes.c_int),
        ('totalConstMem', ctypes.c_size_t),
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int),
        ('textureAlignment', ctypes.c_size_t),
        ('texturePitchAlignment', ctypes.c_size_t),
        ('deviceOverlap', ctypes.c_int),
        ('multiProcessorCount', ctypes.c_int),
        ('kernelExecTimeoutEnabled', ctypes.c_int),
        ('integrated', ctypes.c_int),
        ('canMapHostMemory', ctypes.c_int),
        ('computeMode', ctypes.c_int),
        ('maxTexture1D', ctypes.c_int),
        ('maxTexture1DMipmap', ctypes.c_int),
        ('maxTexture1DLinear', ctypes.c_int),
        ('maxTexture2D', ctypes.c_int * 2),
        ('maxTexture2DMipmap', ctypes.c_int * 2),
        ('maxTexture2DLinear', ctypes.c_int * 3),
        ('maxTexture2DGather', ctypes.c_int * 2),
        ('maxTexture3D', ctypes.c_int * 3),
        ('maxTexture3DAlt', ctypes.c_int * 3),
        ('maxTextureCubemap', ctypes.c_int),
        ('maxTexture1DLayered', ctypes.c_int * 2),
        ('maxTexture2DLayered', ctypes.c_int * 3),
        ('maxTextureCubemapLayered', ctypes.c_int * 2),
        ('maxSurface1D', ctypes.c_int),
        ('maxSurface2D', ctypes.c_int * 2),
        ('maxSurface3D', ctypes.c_int * 3),
        ('maxSurface1DLayered', ctypes.c_int * 2),
        ('maxSurface2DLayered', ctypes.c_int * 3),
        ('maxSurfaceCubemap', ctypes.c_int),
        ('maxSurfaceCubemapLayered', ctypes.c_int * 2),
        ('surfaceAlignment', ctypes.c_size_t),
        ('concurrentKernels', ctypes.c_int),
        ('ECCEnabled', ctypes.c_int),
        ('pciBusID', ctypes.c_int),
        ('pciDeviceID', ctypes.c_int),
        ('pciDomainID', ctypes.c_int),
        ('tccDriver', ctypes.c_int),
        ('asyncEngineCount', ctypes.c_int),
        ('unifiedAddressing', ctypes.c_int),
        ('memoryClockRate', ctypes.c_int),
        ('memoryBusWidth', ctypes.c_int),
        ('l2CacheSize', ctypes.c_int),
        ('maxThreadsPerMultiProcessor', ctypes.c_int),
        ('streamPrioritiesSupported', ctypes.c_int),
        ('globalL1CacheSupported', ctypes.c_int),
        ('localL1CacheSupported', ctypes.c_int),
        ('sharedMemPerMultiprocessor', ctypes.c_size_t),
        ('regsPerMultiprocessor', ctypes.c_int),
        ('managedMemSupported', ctypes.c_int),
        ('isMultiGpuBoard', ctypes.c_int),
        ('multiGpuBoardGroupID', ctypes.c_int),
        # Extra space for new fields in future toolkits
        ('__future_buffer', ctypes.c_int * 128),
        # added later with cudart.cudaDeviceGetPCIBusId
        # (needed by NVML)
        ('pciBusID_str', ctypes.c_char * 16),
    ]


class struct_c_nvmlDevice_t(ctypes.Structure):
    """
    Handle to a device in NVML
    """
    pass  # opaque handle
c_nvmlDevice_t = ctypes.POINTER(struct_c_nvmlDevice_t)


class c_nvmlMemory_t(ctypes.Structure):
    """
    Passed to nvml.nvmlDeviceGetMemoryInfo()
    """
    _fields_ = [
        ('total', ctypes.c_ulonglong),
        ('free', ctypes.c_ulonglong),
        ('used', ctypes.c_ulonglong),
        # Extra space for new fields in future toolkits
        ('__future_buffer', ctypes.c_ulonglong * 8),
    ]


class c_nvmlUtilization_t(ctypes.Structure):
    """
    Passed to nvml.nvmlDeviceGetUtilizationRates()
    """
    _fields_ = [
        ('gpu', ctypes.c_uint),
        ('memory', ctypes.c_uint),
        # Extra space for new fields in future toolkits
        ('__future_buffer', ctypes.c_uint * 8),
    ]


def get_library(name):
    """
    Returns a ctypes.CDLL or None
    """
    try:
        if platform.system() == 'Windows':
            return ctypes.windll.LoadLibrary(name)
        else:
            return ctypes.cdll.LoadLibrary(name)
    except OSError:
        pass
    return None


def get_cudart():
    """
    Return the ctypes.DLL object for cudart or None
    """
    if platform.system() == 'Windows':
        arch = platform.architecture()[0]
        for ver in range(90, 50, -5):
            cudart = get_library('cudart%s_%d.dll' % (arch[:2], ver))
            if cudart is not None:
                return cudart
    elif platform.system() == 'Darwin':
        for major in xrange(9, 5, -1):
            for minor in (5, 0):
                cudart = get_library('libcudart.%d.%d.dylib' % (major, minor))
                if cudart is not None:
                    return cudart
        return get_library('libcudart.dylib')
    else:
        for major in xrange(9, 5, -1):
            for minor in (5, 0):
                cudart = get_library('libcudart.so.%d.%d' % (major, minor))
                if cudart is not None:
                    return cudart
        return get_library('libcudart.so')
    return None


def get_nvml():
    """
    Return the ctypes.DLL object for cudart or None
    """
    if platform.system() == 'Windows':
        return get_library('nvml.dll')
    else:
        for name in (
                'libnvidia-ml.so.1',
                'libnvidia-ml.so',
                'nvml.so'):
            nvml = get_library(name)
            if nvml is not None:
                return nvml
    return None

devices = None


def get_devices(force_reload=False):
    """
    Returns a list of c_cudaDeviceProp's
    Prints an error and returns None if something goes wrong
    Keyword arguments:
    force_reload -- if False, return the previously loaded list of devices
    """
    global devices
    if not force_reload and devices is not None:
        # Only query CUDA once
        return devices
    devices = []

    cudart = get_cudart()
    if cudart is None:
        return []

    # check CUDA version
    cuda_version = ctypes.c_int()
    rc = cudart.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
    if rc != 0:
        logger.error('cudaRuntimeGetVersion() failed with error #%s' % rc)
        return []
    if cuda_version.value < 6050:
        logger.error('ERROR: Cuda version must be >= 6.5, not "%s"' % cuda_version.valu)
        return []

    # get number of devices
    num_devices = ctypes.c_int()
    rc = cudart.cudaGetDeviceCount(ctypes.byref(num_devices))
    if rc != 0:
        logger.error('cudaGetDeviceCount() failed with error #%s' % rc)
        return []

    # query devices
    for x in xrange(num_devices.value):
        properties = c_cudaDeviceProp()
        rc = cudart.cudaGetDeviceProperties(ctypes.byref(properties), x)
        if rc == 0:
            pciBusID_str = ' ' * 16
            # also save the string representation of the PCI bus ID
            rc = cudart.cudaDeviceGetPCIBusId(ctypes.c_char_p(pciBusID_str), 16, x)
            if rc == 0:
                properties.pciBusID_str = pciBusID_str
            devices.append(properties)
        else:
            logger.error('cudaGetDeviceProperties() failed with error #%s' % rc)
        del properties
    return devices


def get_device(device_id):
    """
    Returns a c_cudaDeviceProp
    """
    return get_devices()[int(device_id)]


def get_nvml_info(device_id):
    """
    Gets info from NVML for the given device
    Returns a dict of dicts from different NVML functions
    """
    device = get_device(device_id)
    if device is None:
        return None

    nvml = get_nvml()
    if nvml is None:
        return None

    rc = nvml.nvmlInit()
    if rc != 0:
        raise RuntimeError('nvmlInit() failed with error #%s' % rc)

    try:
        # get device handle
        handle = c_nvmlDevice_t()
        rc = nvml.nvmlDeviceGetHandleByPciBusId(ctypes.c_char_p(device.pciBusID_str), ctypes.byref(handle))
        if rc != 0:
            raise RuntimeError('nvmlDeviceGetHandleByPciBusId() failed with error #%s' % rc)

        # Grab info for this device from NVML
        info = {
            'minor_number': device_id,
            'product_name': device.name
        }

        uuid = ' ' * 41
        rc = nvml.nvmlDeviceGetUUID(handle, ctypes.c_char_p(uuid), 41)
        if rc == 0:
            info['uuid'] = uuid[:-1]

        temperature = ctypes.c_int()
        rc = nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temperature))
        if rc == 0:
            info['temperature'] = temperature.value

        speed = ctypes.c_uint()
        rc = nvml.nvmlDeviceGetFanSpeed(handle, ctypes.byref(speed))
        if rc == 0:
            info['fan'] = speed.value

        power_draw = ctypes.c_uint()
        rc = nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power_draw))
        if rc == 0:
            info['power_draw'] = power_draw.value

        power_limit = ctypes.c_uint()
        rc = nvml.nvmlDeviceGetPowerManagementLimit(handle, ctypes.byref(power_limit))
        if rc == 0:
            info['power_limit'] = power_limit.value

        memory = c_nvmlMemory_t()
        rc = nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
        if rc == 0:
            info['memory_total'] = memory.total
            info['memory_used'] = memory.used
        utilization = c_nvmlUtilization_t()
        rc = nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization))
        if rc == 0:
            info['gpu_util'] = utilization.gpu

        return info
    finally:
        rc = nvml.nvmlShutdown()
        if rc != 0:
            pass


def add_unit(data):
    temperature = 'temperature'
    if temperature in data:
        data[temperature] = '{} C'.format(data[temperature])

    fan = 'fan'
    if fan in data:
        data[fan] = '{} %'.format(data[fan])

    power_draw = 'power_draw'
    if power_draw in data:
        data[power_draw] = '{:.2f} W'.format(float(data[power_draw]) / pow(10, 3))

    power_limit = 'power_limit'
    if power_limit in data:
        data[power_limit] = '{:.2f} W'.format(float(data[power_limit]) / pow(10, 3))

    memory_total = 'memory_total'
    if memory_total in data:
        data[memory_total] = '{} MiB'.format(data[memory_total] / pow(2, 20))

    memory_used = 'memory_used'
    if memory_used in data:
        data[memory_used] = '{} MiB'.format(data[memory_used] / pow(2, 20))

    gpu_util = 'gpu_util'
    if gpu_util in data:
        data[gpu_util] = '{} %'.format(data[gpu_util])


def get_devices_info():
    if not len(get_devices()):
        return None

    nvml = get_nvml()
    nvml.nvmlInit()
    version = ' ' * 80
    nvml.nvmlSystemGetDriverVersion(ctypes.c_char_p(version), 80)
    version = version.strip()[:-1]
    gpus = []

    for i, device in enumerate(get_devices()):
        info = get_nvml_info(i)
        if info:
            gpus.append(info)

    for gpu in gpus:
        add_unit(gpu)

    return {
        'gpus': gpus,
        'driver_version': version
    }
