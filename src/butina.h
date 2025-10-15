#ifndef NVMOLKIT_BUTINA_H
#define NVMOLKIT_BUTINA_H

#include "device_vector.h"
namespace nvMolKit {

void butinaGpu(const AsyncDeviceVector<double>& distanceMatrix,
               AsyncDeviceVector<int>&          clusters,
               double                           cutoff,
               cudaStream_t                     stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLIT_BUTINA_H