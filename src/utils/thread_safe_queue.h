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

#ifndef NVMOLKIT_THREAD_SAFE_QUEUE_H
#define NVMOLKIT_THREAD_SAFE_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace nvMolKit {

/**
 * @brief A thread-safe queue supporting multiple producers and consumers.
 *
 * Provides blocking and non-blocking operations with graceful shutdown semantics.
 * Can be used for work queues (items flow through once) or resource pools
 * (items cycle back via push after pop).
 *
 * @tparam T The type of elements stored in the queue. Must be movable.
 */
template <typename T> class ThreadSafeQueue {
 public:
  ThreadSafeQueue() = default;

  ThreadSafeQueue(const ThreadSafeQueue&)            = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
  ThreadSafeQueue(ThreadSafeQueue&&)                 = delete;
  ThreadSafeQueue& operator=(ThreadSafeQueue&&)      = delete;

  /**
   * @brief Push an item onto the queue.
   *
   * Thread-safe. Notifies one waiting consumer.
   * If the queue is closed, the item is silently dropped.
   *
   * @param item The item to push (moved into the queue)
   */
  void push(T item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;
      }
      queue_.push(std::move(item));
    }
    cv_.notify_one();
  }

  /**
   * @brief Push multiple items onto the queue.
   *
   * Thread-safe. Notifies all waiting consumers.
   * If the queue is closed, items are silently dropped.
   *
   * @tparam Container A container type with begin()/end() iterators
   * @param items The items to push (each is moved into the queue)
   */
  template <typename Container> void pushBatch(Container&& items) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;
      }
      for (auto& item : items) {
        queue_.push(std::move(item));
      }
    }
    cv_.notify_all();
  }

  /**
   * @brief Pop an item from the queue, blocking if empty.
   *
   * Blocks until an item is available or the queue is closed.
   *
   * @return The popped item, or std::nullopt if the queue is closed and empty
   */
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return std::nullopt;
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  /**
   * @brief Try to pop an item without blocking.
   *
   * @return The popped item, or std::nullopt if the queue is empty
   */
  std::optional<T> tryPop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  /**
   * @brief Close the queue, signaling consumers to exit.
   *
   * After closing:
   * - push() and pushBatch() silently drop items
   * - Blocked pop() calls return std::nullopt once queue is empty
   * - New pop() calls return remaining items, then std::nullopt
   */
  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
  }

  /**
   * @brief Get the current number of items in the queue.
   *
   * Note: The returned value may be stale by the time it's used.
   */
  [[nodiscard]] size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /**
   * @brief Check if the queue is empty.
   *
   * Note: The returned value may be stale by the time it's used.
   */
  [[nodiscard]] bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

 private:
  std::queue<T>           queue_;
  mutable std::mutex      mutex_;
  std::condition_variable cv_;
  bool                    closed_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_THREAD_SAFE_QUEUE_H
