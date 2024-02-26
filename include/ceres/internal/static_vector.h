// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: darius.rueckert@fau.de (Darius Rueckert)
//

#ifndef CERES_PUBLIC_INTERNAL_STATIC_VECTOR_H_
#define CERES_PUBLIC_INTERNAL_STATIC_VECTOR_H_

#include <glog/logging.h>

#include <array>
#include <vector>

namespace ceres::internal {

template <typename Tp, std::size_t Nm>
class CERES_NO_EXPORT StaticVector {
 public:
  typedef Tp* iterator;
  typedef const Tp* const_iterator;

  void push_back(const Tp& x) {
    CHECK(mSize < Nm);
    mData[mSize] = x;
    mSize++;
  }

  Tp& at(std::size_t aIndex) { return mData[aIndex]; }

  const Tp& at(std::size_t aIndex) const { return mData[aIndex]; }

  Tp* data() { return mData; }

  const Tp* data() const { return mData; }

  std::size_t size() const { return mSize; }

  Tp& operator[](std::size_t aIndex) { return mData[aIndex]; }

  const Tp& operator[](std::size_t aIndex) const { return mData[aIndex]; }

  bool empty() const { return mSize == 0; }

  iterator begin() noexcept { return iterator(data()); }

  const_iterator begin() const noexcept { return const_iterator(data()); }

  iterator end() noexcept { return iterator(data() + mSize); }

  const_iterator end() const noexcept { return const_iterator(data() + mSize); }

 private:
  std::size_t mSize{0};
  Tp mData[Nm];
};

template <typename _Tp, std::size_t _Nm>
inline bool operator==(const StaticVector<_Tp, _Nm>& __one,
                       const StaticVector<_Tp, _Nm>& __two) {
  return std::equal(__one.begin(), __one.end(), __two.begin());
}

}  // namespace ceres::internal

#endif  // CERES_PUBLIC_INTERNAL_ARRAY_SELECTOR_H_
