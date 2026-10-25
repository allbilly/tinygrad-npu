# generated from RK3588 rknpu_register.h; minimal register subset used by the clean backend
import ctypes, functools

def _ioctl(nr, payload_type, fd, payload=None, **kwargs):
  made = payload or payload_type(**kwargs)
  ret = fd.ioctl((3 << 30) | (ctypes.sizeof(made) << 16) | (ord('d') << 8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

class struct_rknpu_mem_create(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('handle', ctypes.c_uint32), ('flags', ctypes.c_uint32), ('size', ctypes.c_uint64), ('obj_addr', ctypes.c_uint64),
              ('dma_addr', ctypes.c_uint64), ('sram_size', ctypes.c_uint64)]
class struct_rknpu_mem_map(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('handle', ctypes.c_uint32), ('reserved', ctypes.c_uint32), ('offset', ctypes.c_uint64)]
class struct_rknpu_mem_destroy(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('handle', ctypes.c_uint32), ('reserved', ctypes.c_uint32), ('obj_addr', ctypes.c_uint64)]
class struct_rknpu_mem_sync(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('flags', ctypes.c_uint32), ('reserved', ctypes.c_uint32), ('obj_addr', ctypes.c_uint64),
              ('offset', ctypes.c_uint64), ('size', ctypes.c_uint64)]
class struct_rknpu_task(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('flags', ctypes.c_uint32), ('op_idx', ctypes.c_uint32), ('enable_mask', ctypes.c_uint32), ('int_mask', ctypes.c_uint32),
              ('int_clear', ctypes.c_uint32), ('int_status', ctypes.c_uint32), ('regcfg_amount', ctypes.c_uint32),
              ('regcfg_offset', ctypes.c_uint32), ('regcmd_addr', ctypes.c_uint64)]
class struct_rknpu_subcore_task(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('task_start', ctypes.c_uint32), ('task_number', ctypes.c_uint32)]
class struct_rknpu_submit(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('flags', ctypes.c_uint32), ('timeout', ctypes.c_uint32), ('task_start', ctypes.c_uint32), ('task_number', ctypes.c_uint32),
              ('task_counter', ctypes.c_uint32), ('priority', ctypes.c_int32), ('task_obj_addr', ctypes.c_uint64),
              ('regcfg_obj_addr', ctypes.c_uint64), ('task_base_addr', ctypes.c_uint64), ('user_data', ctypes.c_uint64),
              ('core_mask', ctypes.c_uint32), ('fence_fd', ctypes.c_int32), ('subcore_task', struct_rknpu_subcore_task * 5)]
class struct_rknpu_action(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('flags', ctypes.c_uint32), ('value', ctypes.c_uint32)]

DRM_IOCTL_RKNPU_ACTION = functools.partial(_ioctl, 0x40, struct_rknpu_action)
DRM_IOCTL_RKNPU_SUBMIT = functools.partial(_ioctl, 0x41, struct_rknpu_submit)
DRM_IOCTL_RKNPU_MEM_CREATE = functools.partial(_ioctl, 0x42, struct_rknpu_mem_create)
DRM_IOCTL_RKNPU_MEM_MAP = functools.partial(_ioctl, 0x43, struct_rknpu_mem_map)
DRM_IOCTL_RKNPU_MEM_DESTROY = functools.partial(_ioctl, 0x44, struct_rknpu_mem_destroy)
DRM_IOCTL_RKNPU_MEM_SYNC = functools.partial(_ioctl, 0x45, struct_rknpu_mem_sync)
RKNPU_MEM_NON_CACHEABLE, RKNPU_MEM_NON_CONTIGUOUS, RKNPU_MEM_CACHEABLE = 0, 1, 2
RKNPU_MEM_KERNEL_MAPPING, RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT = 8, 1024
RKNPU_MEM_SYNC_TO_DEVICE, RKNPU_MEM_SYNC_FROM_DEVICE = 1, 2
RKNPU_JOB_PC, RKNPU_JOB_BLOCK, RKNPU_JOB_PINGPONG, RKNPU_ACT_RESET = 1, 0, 4, 6

REG_PC_OPERATION_ENABLE = 0x00000008
REG_PC_BASE_ADDRESS = 0x00000010
REG_PC_REGISTER_AMOUNTS = 0x00000014
TARGET_PC, TARGET_PC_REG, TARGET_VERSION = 0x0081, 0x0101, 0x0041
REG_DPU_S_POINTER = 0x00004004
REG_DPU_FEATURE_MODE_CFG = 0x0000400c
REG_DPU_DATA_FORMAT = 0x00004010
REG_DPU_DST_BASE_ADDR = 0x00004020
REG_DPU_DST_SURF_STRIDE = 0x00004024
REG_DPU_DATA_CUBE_WIDTH = 0x00004030
REG_DPU_DATA_CUBE_HEIGHT = 0x00004034
REG_DPU_DATA_CUBE_NOTCH_ADDR = 0x00004038
REG_DPU_DATA_CUBE_CHANNEL = 0x0000403c
REG_DPU_BS_CFG = 0x00004040
REG_DPU_BS_ALU_CFG, REG_DPU_BS_MUL_CFG = 0x00004044, 0x00004048
REG_DPU_BS_OW_CFG = 0x00004050
REG_DPU_WDMA_SIZE_0 = 0x00004058
REG_DPU_WDMA_SIZE_1 = 0x0000405c
REG_DPU_BN_CFG = 0x00004060
REG_DPU_BN_ALU_CFG = 0x00004064
REG_DPU_BN_MUL_CFG = 0x00004068
REG_DPU_BN_RELUX_CMP_VALUE = 0x0000406c
REG_DPU_EW_CFG = 0x00004070
REG_DPU_EW_CVT_SCALE_VALUE = 0x00004078
REG_DPU_EW_RELUX_CMP_VALUE = 0x0000407c
REG_DPU_OUT_CVT_OFFSET = 0x00004080
REG_DPU_OUT_CVT_SCALE = 0x00004084
REG_DPU_OUT_CVT_SHIFT = 0x00004088
REG_DPU_SURFACE_ADD = 0x000040c0
REG_DPU_LUT_ACCESS_CFG, REG_DPU_LUT_ACCESS_DATA = 0x00004100, 0x00004104
REG_DPU_LUT_CFG, REG_DPU_LUT_INFO = 0x00004108, 0x0000410c
REG_DPU_LUT_LE_START, REG_DPU_LUT_LE_END = 0x00004110, 0x00004114
REG_DPU_LUT_LO_START, REG_DPU_LUT_LO_END = 0x00004118, 0x0000411c
REG_DPU_LUT_LO_SLOPE_SCALE, REG_DPU_LUT_LO_SLOPE_SHIFT = 0x00004128, 0x0000412c
REG_DPU_RDMA_RDMA_S_POINTER = 0x00005004
REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH = 0x0000500c
REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT = 0x00005010
REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL = 0x00005014
REG_DPU_RDMA_RDMA_SRC_BASE_ADDR = 0x00005018
REG_DPU_RDMA_RDMA_ERDMA_CFG = 0x00005034
REG_DPU_RDMA_RDMA_EW_BASE_ADDR = 0x00005038
REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG = 0x00005044
REG_DPU_RDMA_RDMA_WEIGHT = 0x00005068
REG_CNA_CONV_CON1, REG_CNA_CONV_CON2, REG_CNA_CONV_CON3 = 0x100c, 0x1010, 0x1014
REG_CNA_DATA_SIZE0, REG_CNA_DATA_SIZE1, REG_CNA_DATA_SIZE2, REG_CNA_DATA_SIZE3 = 0x1020, 0x1024, 0x1028, 0x102c
REG_CNA_WEIGHT_SIZE0, REG_CNA_WEIGHT_SIZE1, REG_CNA_WEIGHT_SIZE2 = 0x1030, 0x1034, 0x1038
REG_CNA_CBUF_CON0, REG_CNA_CBUF_CON1 = 0x1040, 0x1044
REG_CNA_CVT_CON0, REG_CNA_CVT_CON1, REG_CNA_CVT_CON2 = 0x104c, 0x1050, 0x1054
REG_CNA_CVT_CON3, REG_CNA_CVT_CON4 = 0x1058, 0x105c
REG_CNA_FEATURE_DATA_ADDR, REG_CNA_DMA_CON0, REG_CNA_DMA_CON1, REG_CNA_DMA_CON2 = 0x1070, 0x1078, 0x107c, 0x1080
REG_CNA_FC_DATA_SIZE0, REG_CNA_FC_DATA_SIZE1, REG_CNA_DCOMP_ADDR0 = 0x1084, 0x1088, 0x1110
REG_CORE_MISC_CFG, REG_CORE_DATAOUT_SIZE_0, REG_CORE_DATAOUT_SIZE_1, REG_CORE_RESERVED_3030 = 0x3010, 0x3014, 0x3018, 0x3030
REG_PPU_S_POINTER, REG_PPU_DATA_CUBE_IN_WIDTH, REG_PPU_DATA_CUBE_IN_HEIGHT = 0x6004, 0x600c, 0x6010
REG_PPU_DATA_CUBE_IN_CHANNEL, REG_PPU_DATA_CUBE_OUT_WIDTH, REG_PPU_DATA_CUBE_OUT_HEIGHT = 0x6014, 0x6018, 0x601c
REG_PPU_DATA_CUBE_OUT_CHANNEL, REG_PPU_OPERATION_MODE_CFG, REG_PPU_POOLING_KERNEL_CFG = 0x6020, 0x6024, 0x6034
REG_PPU_RECIP_KERNEL_WIDTH, REG_PPU_RECIP_KERNEL_HEIGHT = 0x6038, 0x603c
REG_PPU_DST_BASE_ADDR, REG_PPU_DST_SURF_STRIDE, REG_PPU_DATA_FORMAT, REG_PPU_MISC_CTRL = 0x6070, 0x607c, 0x6084, 0x60dc
REG_PPU_RDMA_RDMA_S_POINTER, REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH = 0x7004, 0x700c
REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT, REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL = 0x7010, 0x7014
REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE = 0x701c, 0x7024
REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, REG_PPU_RDMA_RDMA_DATA_FORMAT = 0x7028, 0x7030
