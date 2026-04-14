// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_MMFF_PYTHON_UTILS_H
#define NVMOLKIT_MMFF_PYTHON_UTILS_H

#include <boost/python.hpp>
#include <vector>

#include "mmff_properties.h"

namespace nvMolKit {

inline MMFFProperties extractMMFFProperties(const boost::python::object& obj,
                                            double                       nonBondedThreshold          = 100.0,
                                            bool                         ignoreInterfragInteractions = true) {
  MMFFProperties props;
  if (obj.is_none()) {
    props.nonBondedThreshold          = nonBondedThreshold;
    props.ignoreInterfragInteractions = ignoreInterfragInteractions;
    return props;
  }
  props = boost::python::extract<MMFFProperties>(obj);
  return props;
}

inline std::vector<MMFFProperties> extractMMFFPropertiesList(const boost::python::list& properties, int numMols) {
  const int                   n = boost::python::len(properties);
  std::vector<MMFFProperties> props;
  props.reserve(numMols);
  for (int i = 0; i < numMols; ++i) {
    if (i < n) {
      props.push_back(extractMMFFProperties(boost::python::object(properties[i])));
    } else {
      props.emplace_back();
    }
  }
  return props;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_PYTHON_UTILS_H
