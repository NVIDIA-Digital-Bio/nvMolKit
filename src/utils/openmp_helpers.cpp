#include "openmp_helpers.h"

namespace nvMolKit {
namespace detail {

void OpenMPExceptionRegistry::store(std::exception_ptr exceptionPtr) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!exception_) {
    exception_ = std::move(exceptionPtr);
  }
}

void OpenMPExceptionRegistry::rethrow() {
  std::exception_ptr toThrow;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    toThrow    = exception_;
    exception_ = nullptr;
  }

  if (toThrow) {
    std::rethrow_exception(toThrow);
  }
}

}  // namespace detail
}  // namespace nvMolKit