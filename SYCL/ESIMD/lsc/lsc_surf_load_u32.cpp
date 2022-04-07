//==------- lsc_surf_load_u32.cpp - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_surf_load.hpp"

constexpr uint32_t seed = 199;

int main(void) {
  srand(seed);
  bool passed = true;

  // non transpose
  passed &= test<0, uint32_t, 1, 4, 32, 1, false>(rand());
  passed &= test<1, uint32_t, 1, 4, 32, 2, false>(rand());
  passed &= test<2, uint32_t, 1, 4, 16, 2, false>(rand());
  passed &= test<3, uint32_t, 1, 4, 4, 1, false>(rand());
  passed &= test<4, uint32_t, 1, 1, 1, 1, false>(1);
  passed &= test<5, uint32_t, 2, 1, 1, 1, false>(1);

  // passed &= test<6, uint32_t, 1, 4, 8, 2, false>(rand());
  // passed &= test<7, uint32_t, 1, 4, 8, 3, false>(rand());

  // transpose
  passed &= test<8, uint32_t, 1, 4, 1, 32, true>();
  passed &= test<9, uint32_t, 2, 2, 1, 16, true>();
  passed &= test<10, uint32_t, 4, 4, 1, 4, true>();

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
