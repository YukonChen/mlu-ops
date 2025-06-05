#include "dot_product.h"

#include <iostream>
#include <string>

#include "core/runtime/device.h"

#define op_name "[mluOpDotProduct]"

mluOpStatus_t MLUOP_WIN_API mluOpDotProduct(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc, const void *x,
    const mluOpTensorDescriptor_t y_desc, const void *y,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  // Check if handle, descriptors, and data pointers are valid
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, x_desc != NULL);
  PARAM_CHECK(op_name, y_desc != NULL);
  PARAM_CHECK(op_name, output_desc != NULL);

  // Check if data types are compatible
  if (x_desc->getDtype() != y_desc->getDtype() ||
      x_desc->getDtype() != output_desc->getDtype()) {
    LOG(ERROR) << op_name << ": Data types of x, y, and output must match.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // Check if dimensions are compatible
  if (x_desc->getDim() != y_desc->getDim()) {
    LOG(ERROR) << op_name << ": Dimensions of x, y must match.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < x_desc->getDim(); ++i) {
    if (x_desc->getDimIndex(i) != y_desc->getDimIndex(i)) {
      LOG(ERROR) << op_name << ": Shapes of x, y must match.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  // Check 0 element tensors
  const int32_t element_num = mluOpGetTensorElementNum(x_desc);
  if (element_num == 0) {
    VLOG(5) << op_name << ": Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  // Check if data pointers are not null
  PARAM_CHECK(op_name, x != NULL);
  PARAM_CHECK(op_name, y != NULL);
  PARAM_CHECK(op_name, output != NULL);

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  size_t union_number = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_in_cluster = handle->core_num_per_cluster;
  k_type = cnrtFuncTypeUnion1;
  k_dim.x = core_in_cluster;
  k_dim.y = union_number;
  k_dim.z = 1;
  mluOpDataType_t k_datatype = x_desc->getDtype();

  // Launch the kernel for dot product
  MLUOP_CHECK(KernelDotProduct(k_dim, k_type, handle->queue, k_datatype, x, y,
                               output, element_num));
  return MLUOP_STATUS_SUCCESS;
}