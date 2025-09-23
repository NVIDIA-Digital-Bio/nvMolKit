// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <DataStructs/ExplicitBitVect.h>

#include <boost/python.hpp>

#include "array_helpers.h"
#include "similarity.h"

template <typename T> boost::python::list vectorToList(const std::vector<T>& vec) {
  boost::python::list list;
  for (const auto& value : vec) {
    list.append(value);
  }
  return list;
}

namespace {

template <typename blockT> cuda::std::span<const blockT> getSpanFromDictElems(void* data, boost::python::tuple& shape) {
  int size = boost::python::extract<int>(shape[0]);
  // multiply by any other dimensions
  for (int i = 1; i < len(shape); i++) {
    size *= boost::python::extract<int>(shape[i]);
  }

  return cuda::std::span<blockT>(reinterpret_cast<blockT*>(data), size);
}

}  // namespace

BOOST_PYTHON_MODULE(_DataStructs) {
  boost::python::def(
    "CrossTanimotoSimilarityRawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo) {
      // Extract boost::python::tuple from dict['shape']
      boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
      boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

      const int nInts    = boost::python::extract<int>(shapeOne[1]);
      const int nIntsTwo = boost::python::extract<int>(shapeTwo[1]);
      if (nInts != nIntsTwo) {
        throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
      }
      const size_t numMolsOne = boost::python::extract<int>(shapeOne[0]);
      const size_t numMolsTwo = boost::python::extract<int>(shapeTwo[0]);

      // Extract the datatype string, and check the number of bytes
      const int nBytes = sizeof(std::uint32_t);

      const int            fpSize       = nInts * 8 * nBytes;
      boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
      size_t               data1Pointer = boost::python::extract<std::size_t>(data1[0]);
      boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
      size_t               data2Pointer = boost::python::extract<std::size_t>(data2[0]);

      auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
      auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);
      auto array = nvMolKit::crossTanimotoSimilarityGpuResult(span1, span2, fpSize);
      assert(array.size() == numMolsOne * numMolsTwo);
      return makePyArray(array, boost::python::make_tuple(numMolsOne, numMolsTwo));
    },
    boost::python::return_value_policy<boost::python::manage_new_object>());

  // --------------------------------
  // Cosine similarity binding functions
  // --------------------------------

  boost::python::def(
    "CrossCosineSimilarityRawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo) {
      // Extract boost::python::tuple from dict['shape']
      boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
      boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

      const int nInts    = boost::python::extract<int>(shapeOne[1]);
      const int nIntsTwo = boost::python::extract<int>(shapeTwo[1]);
      if (nInts != nIntsTwo) {
        throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
      }
      const size_t numMolsOne = boost::python::extract<int>(shapeOne[0]);
      const size_t numMolsTwo = boost::python::extract<int>(shapeTwo[0]);

      // Extract the datatype string, and check the number of bytes
      const int nBytes = sizeof(std::uint32_t);

      const int            fpSize       = nInts * 8 * nBytes;
      boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
      size_t               data1Pointer = boost::python::extract<std::size_t>(data1[0]);
      boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
      size_t               data2Pointer = boost::python::extract<std::size_t>(data2[0]);

      auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
      auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);
      auto array = nvMolKit::crossCosineSimilarityGpuResult(span1, span2, fpSize);
      assert(array.size() == numMolsOne * numMolsTwo);
      return makePyArray(array, boost::python::make_tuple(numMolsOne, numMolsTwo));
    },
    boost::python::return_value_policy<boost::python::manage_new_object>());
}
