#ifndef KERNELS_DOT_PRODUCT_DOT_PRODUCT_H
#define KERNELS_DOT_PRODUCT_DOT_PRODUCT_H

#include "mlu_op.h"

mluOpStatus_t MLUOP_WIN_API KernelDotProduct(cnrtDim3_t k_dim,
                                             cnrtFunctionType_t k_type,
                                             cnrtQueue_t queue,
                                             mluOpDataType_t d_type,
                                             const void *x, const void *y,
                                             void *output, size_t element_num);

#endif  // KERNELS_DOT_PRODUCT_DOT_PRODUCT_H