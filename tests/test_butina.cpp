#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "butina.h"
#include "device.h"
#include "host_vector.h"

using nvMolKit::AsyncDeviceVector;

static std::vector<double> tenX10Distances = {
  3.93520250e-01, 9.72041439e-01, 7.06006593e-01, 9.50663718e-01, 6.67823709e-01, 9.89277498e-01, 6.19711942e-02,
  2.00787742e-01, 7.78239377e-01, 9.38741376e-01, 1.82373283e-01, 4.48975814e-01, 2.68622302e-02, 3.35987923e-01,
  2.41733963e-01, 4.53879633e-01, 3.19503452e-01, 9.62324019e-02, 4.09404330e-01, 5.16189668e-01, 7.03719124e-01,
  2.89864581e-02, 1.00685250e-01, 3.40350691e-01, 2.20363260e-01, 6.95868803e-01, 2.16477890e-01, 7.94510812e-01,
  9.79031234e-02, 8.57265980e-01, 5.12496789e-01, 4.95047222e-01, 8.15341947e-01, 4.87856187e-01, 4.31629275e-01,
  5.82766898e-02, 7.64265558e-01, 6.43155558e-01, 3.19877317e-01, 2.78333991e-01, 2.77955377e-01, 6.64189879e-01,
  9.35921431e-01, 5.16564153e-01, 2.37088457e-01, 8.51908719e-01, 9.46063616e-01, 5.11695791e-01, 6.54142657e-01,
  6.38641018e-01, 4.47187705e-01, 9.36425958e-01, 7.70492073e-01, 1.85065105e-01, 5.85568862e-01, 1.15888133e-01,
  3.09956150e-01, 8.35447043e-01, 1.30865852e-01, 2.80182338e-01, 9.07094110e-01, 6.21056958e-01, 4.90428752e-02,
  2.08698134e-01, 4.04065907e-01, 4.92263349e-01, 7.11524930e-01, 9.02871102e-01, 3.81285105e-01, 1.87138721e-01,
  3.20067466e-01, 5.48462783e-01, 9.65866634e-02, 9.93503855e-01, 3.68132345e-01, 4.47171861e-01, 7.32548263e-01,
  1.36630516e-01, 7.31898534e-03, 4.48492346e-01, 3.29984727e-01, 8.28669668e-02, 3.72850568e-01, 3.37745905e-01,
  2.61022903e-01, 8.73056532e-01, 3.10624522e-01, 9.08376835e-01, 2.45440403e-02, 9.32461320e-02, 1.50251270e-01,
  7.21490661e-01, 2.49309682e-04, 6.46522949e-01, 9.06069227e-01, 4.67102359e-01, 8.10106559e-01, 2.01373846e-01,
  9.68380418e-01, 4.01053272e-01};

TEST(ButinaClusterTest, KnownResult) {
  constexpr int          nPts   = 10;
  constexpr double       cutoff = 0.2;
  nvMolKit::ScopedStream scopedStream;
  cudaStream_t           stream = scopedStream.stream();

  AsyncDeviceVector<double> distancesDev(tenX10Distances.size(), stream);
  AsyncDeviceVector<int>    resultDev(nPts, stream);
  distancesDev.copyFromHost(tenX10Distances);

  nvMolKit::butinaGpu(distancesDev, resultDev, cutoff, stream);
  std::vector<int> got(nPts);
  resultDev.copyToHost(got);
  cudaStreamSynchronize(stream);
  std::vector<int> want = {4, 0, 1, 2, 5, 2, 3, 1, 0, 0};
  EXPECT_THAT(got, ::testing::ElementsAreArray(want));
}