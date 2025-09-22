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
#include <boost/python/numpy.hpp>
#include <memory>
#include <stdexcept>

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
  boost::python::numpy::initialize();
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

  // --------------------------------
  // CPU-result similarity binding functions (no options exposed; nullopt by default)
  // --------------------------------

  boost::python::def(
    "CrossTanimotoSimilarityCPURawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo) {
      boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
      boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

      const size_t numMolsOne = boost::python::extract<int>(shapeOne[0]);
      const size_t numMolsTwo = boost::python::extract<int>(shapeTwo[0]);

      const int nInts    = boost::python::extract<int>(shapeOne[1]);
      const int nIntsTwo = boost::python::extract<int>(shapeTwo[1]);
      if (nInts != nIntsTwo) {
        throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
      }

      const int            nBytes       = sizeof(std::uint32_t);
      const int            fpSize       = nInts * 8 * nBytes;
      boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
      size_t               data1Pointer = boost::python::extract<std::size_t>(data1[0]);
      boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
      size_t               data2Pointer = boost::python::extract<std::size_t>(data2[0]);

      auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
      auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);
      auto vec   = nvMolKit::crossTanimotoSimilarityCPUResult(span1, span2, fpSize);

      // Move vector to heap and tie lifetime to a capsule owner
      auto  heapVec = std::make_unique<std::vector<double>>(std::move(vec));
      void* dataPtr = static_cast<void*>(heapVec->data());

      // Capsule destructor to free heapVec
      auto deleter = [](PyObject* capsule) {
        void* ptr = PyCapsule_GetPointer(capsule, "nvmolkit.double_vector");
        auto* v   = reinterpret_cast<std::vector<double>*>(ptr);
        delete v;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(heapVec.get()), "nvmolkit.double_vector", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for CPU similarity result");
      }
      boost::python::object owner{boost::python::handle<>(cap)};
      heapVec.release();

      const Py_intptr_t shape_arr[2]   = {static_cast<Py_intptr_t>(numMolsOne), static_cast<Py_intptr_t>(numMolsTwo)};
      const Py_intptr_t strides_arr[2] = {static_cast<Py_intptr_t>(numMolsTwo * sizeof(double)), static_cast<Py_intptr_t>(sizeof(double))};

      auto arr = boost::python::numpy::from_data(
        dataPtr,
        boost::python::numpy::dtype::get_builtin<double>(),
        boost::python::make_tuple(shape_arr[0], shape_arr[1]),
        boost::python::make_tuple(strides_arr[0], strides_arr[1]),
        owner);
      return arr;
    });

  boost::python::def(
    "CrossCosineSimilarityCPURawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo) {
      boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
      boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

      const size_t numMolsOne = boost::python::extract<int>(shapeOne[0]);
      const size_t numMolsTwo = boost::python::extract<int>(shapeTwo[0]);

      const int nInts    = boost::python::extract<int>(shapeOne[1]);
      const int nIntsTwo = boost::python::extract<int>(shapeTwo[1]);
      if (nInts != nIntsTwo) {
        throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
      }

      const int            nBytes       = sizeof(std::uint32_t);
      const int            fpSize       = nInts * 8 * nBytes;
      boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
      size_t               data1Pointer = boost::python::extract<std::size_t>(data1[0]);
      boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
      size_t               data2Pointer = boost::python::extract<std::size_t>(data2[0]);

      auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
      auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);
      auto vec   = nvMolKit::crossCosineSimilarityCPUResult(span1, span2, fpSize);

      auto  heapVec = std::make_unique<std::vector<double>>(std::move(vec));
      void* dataPtr = static_cast<void*>(heapVec->data());
      auto  deleter = [](PyObject* capsule) {
        void* ptr = PyCapsule_GetPointer(capsule, "nvmolkit.double_vector");
        auto* v   = reinterpret_cast<std::vector<double>*>(ptr);
        delete v;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(heapVec.get()), "nvmolkit.double_vector", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for CPU similarity result");
      }
      boost::python::object owner{boost::python::handle<>(cap)};
      heapVec.release();

      const Py_intptr_t shape_arr[2]   = {static_cast<Py_intptr_t>(numMolsOne), static_cast<Py_intptr_t>(numMolsTwo)};
      const Py_intptr_t strides_arr[2] = {static_cast<Py_intptr_t>(numMolsTwo * sizeof(double)), static_cast<Py_intptr_t>(sizeof(double))};

      auto arr = boost::python::numpy::from_data(
        dataPtr,
        boost::python::numpy::dtype::get_builtin<double>(),
        boost::python::make_tuple(shape_arr[0], shape_arr[1]),
        boost::python::make_tuple(strides_arr[0], strides_arr[1]),
        owner);
      return arr;
    });
}
