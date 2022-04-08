//==------- ctor_converting_fp_extra.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Test for simd converting constructor for extra fp types.
// This test uses extra fp data types with different dimensionality, base and
// step values and different simd constructor invocation contexts.
// The test do the following actions:
//  - construct simd with source data type, then construct simd with destination
//    type from the earlier constructed simd
//  - compare retrieved and expected values

#include "ctor_converting.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

#ifndef ESIMD_TESTS_ENABLE_ALL_DOUBLE_TO_UINT_COMBINATIONS
template <typename T>
using not_double_type = std::bool_constant<!std::is_same_v<double, T>>;
template <typename T> using not_32bit_type = std::bool_constant<sizeof(T) != 4>;
#endif

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto fp_extra_types = get_tested_types<tested_types::fp_extra>();
  const auto fp_types = get_tested_types<tested_types::fp>();
  const auto uint_types = get_tested_types<tested_types::uint>();
  const auto sint_types = get_tested_types<tested_types::sint>();
  const auto core_types = get_tested_types<tested_types::core>();
  const auto single_size = get_sizes<1, 8>();
  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();

  // Run for specific combinations of source types, vector length, destination
  // types and invocation contexts
  passed &= for_all_combinations<ctors::run_test>(fp_extra_types, single_size,
                                                  fp_types, contexts, queue);
  // Note: checks below would expectedly fail with
  // ESIMD_TESTS_ENABLE_HALF_DENORM_MIN_CAST_FROM_FLOAT enabled due to known
  // issue with static_cast value<sycl::half>::denorm_min from float/double to
  // sycl::half
  // Log details: retrieved: 5.9605e-08 [0x1], expected: -0 [0x8000]
  //              after conversion from 5.96046448e-08 [0x33800000]
  passed &= for_all_combinations<ctors::run_test>(
      fp_types, single_size, fp_extra_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(
      fp_extra_types, single_size, fp_extra_types, contexts, queue);
#ifdef ESIMD_TESTS_ENABLE_ALL_DOUBLE_TO_UINT_COMBINATIONS
  passed &= for_all_combinations<ctors::run_test>(fp_extra_types, single_size,
                                                  uint_types, contexts, queue);
#elif defined(ESIMD_TESTS_FULL_COVERAGE)
  // An issue converting from double to unsigned int or unsigned long was
  // observed on win64; both target types are 32-bit for win64 configuration
  // Log details: retrieved: 2147483647, expected: 4294967295
  {
    const auto not_double_types = fp_extra_types.filter_by<not_double_type>();
    const auto not_uint32_types = uint_types.filter_by<not_32bit_type>();
    const auto double_t = named_type_pack<double>::generate("double");

    passed &= for_all_combinations<ctors::run_test>(
        not_double_types, single_size, uint_types, contexts, queue);
    passed &= for_all_combinations<ctors::run_test>(
        double_t, single_size, not_uint32_types, contexts, queue);
  }
#endif
  passed &= for_all_combinations<ctors::run_test>(fp_extra_types, single_size,
                                                  sint_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(uint_types, single_size,
                                                  core_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(sint_types, single_size,
                                                  uint_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(sint_types, single_size,
                                                  sint_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(
      sint_types, single_size, fp_extra_types, contexts, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
