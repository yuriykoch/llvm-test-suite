//===-- value_conv.hpp - This file provides common functions generate values for
//      testing. ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions that let obtain converted reference data for
/// test according to current underlying types.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "logger.hpp"
#include "type_traits.hpp"
#include "value.hpp"

#include <cassert>

namespace esimd_test::api::functional {

// Utility class to retrieve specific values for tests depending on the source
// and destination data types. May be used to retrieve converted reference data.
// All provided methods are safe to use and protected from UB when call
// static_cast<int>(unsigned int).
template <typename SrcT, typename DstT> struct value_conv {
  using less_precision_type =
      std::conditional_t<(value<SrcT>::digits2() < value<DstT>::digits2()),
                         SrcT, DstT>;

  static inline SrcT denorm_min() {
    if constexpr (!type_traits::is_sycl_floating_point_v<SrcT>) {
      // Return zero for any integral type the same way std::denorm_min does
      return 0;
    } else {
      // Return the higher denorm_min value from the type with less precision
      return static_cast<SrcT>(value<less_precision_type>::denorm_min());
    }
  }

  static inline SrcT min() {
    if constexpr (std::is_unsigned_v<SrcT>) {
      // Use zero explicitly
      return 0;
    } else if constexpr (std::is_unsigned_v<DstT> &&
                         type_traits::is_sycl_floating_point_v<SrcT>) {
      // While there is a well-defined value wrap for converting from signed
      // integer type to the unsigned integer type, an attempt to trigger such
      // wrap while converting from the floating point type directly would
      // result in UB according to the C++17
      // So we shouldn't use negative values for such case at all.
      return 0;
    } else {
      // For conversion from signed to signed:
      //   use the value which could be represented exactly in the both source
      //   and destination types
      // For conversion from signed integral to unsigned:
      //   trigger value wrap by using the signed integral source value
      return static_cast<SrcT>(
          value<less_precision_type>::lowest_exact_integral());
    }
  }

  static inline SrcT max() {
    // Use the value which could be represented exactly in the both source and
    // destination types
    return static_cast<SrcT>(value<less_precision_type>::max_exact_integral());
  }
};

// Provides std::vector with the reference data according to the obtained data
// types and number of elements.
template <typename SrcT, typename DstT, int NumElems>
std::vector<SrcT> generate_ref_conv_data() {
  static_assert(std::is_integral_v<SrcT> ||
                    type_traits::is_sycl_floating_point_v<SrcT>,
                "Invalid source type.");
  static_assert(std::is_integral_v<DstT> ||
                    type_traits::is_sycl_floating_point_v<DstT>,
                "Invalid destination type.");

  static const SrcT max = value_conv<SrcT, DstT>::max();
  static const SrcT min = value_conv<SrcT, DstT>::min();
  static const SrcT max_half = max / 2;
  static const SrcT min_half = min / 2;

  std::vector<SrcT> ref_data;

  if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                type_traits::is_sycl_floating_point_v<DstT>) {
    static const SrcT nan = value<SrcT>::nan();
    static const SrcT inf = value<SrcT>::inf();
    static const SrcT denorm = value_conv<SrcT, DstT>::denorm_min();

    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {min, max, -0.0, +0.0, 0.1, denorm, nan, -inf});
  } else if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                       std::is_unsigned_v<DstT>) {
    // We cannot expect negative values to wrap during conversion from
    // the floating point type to the unsigned integral type.
    // The C++17 standard has the following statement:
    // A prvalue of a floating-point type can be converted to a prvalue of an
    // integer type. The conversion truncates; that is, the fractional part is
    // discarded. The behavior is undefined if the truncated value cannot be
    // represented in the destination type.
    ref_data =
        details::construct_ref_data<SrcT, NumElems>({0.0, max, max_half});
  } else if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                       type_traits::is_sycl_signed_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {-0.0, max, max_half, min, min_half});
  } else if constexpr (type_traits::is_sycl_signed_v<SrcT> &&
                       type_traits::is_sycl_signed_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {min, min_half, 0, max_half, max});
  } else if constexpr (type_traits::is_sycl_signed_v<SrcT> &&
                       std::is_unsigned_v<DstT>) {
    static const SrcT src_min = value<SrcT>::lowest();
    static const SrcT src_min_half = src_min / 2;

    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {src_min, src_min_half, 0, max_half, max});
  } else if constexpr (std::is_unsigned_v<SrcT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>({0, max_half, max});
  } else {
    static_assert(!std::is_same_v<SrcT, SrcT>, "Unexpected types combination");
  }

  return ref_data;
}

template <typename DstT> struct static_cast_to {
  template <typename SrcT>
  static bool is_expected_result_for(SrcT reference, DstT retrieved,
                                     DstT &min_expected, DstT &max_expected) {
    // We are safe to assume our reference values generation logic doesn't
    // generate anything out of range
    // [value<DstT>::lowest()...value<DstT>::max()]
    min_expected = static_cast<DstT>(reference);
    max_expected = min_expected;

    static_assert(std::is_reference<decltype(min_expected)>::value);
    static_assert(std::is_reference<decltype((min_expected))>::value);

    if constexpr (!type_traits::is_sycl_floating_point_v<DstT>) {
      // Well-defined for conversion to integral types
      return retrieved == min_expected;
    } else {
      // Handle NaN source value case first
      // We cannot call std::isnan() for integral types because of ambiguous
      // call. Related GitHub issue: https://github.com/microsoft/STL/issues/519
      if constexpr (type_traits::is_sycl_floating_point_v<SrcT>) {
        if (std::isnan(reference)) {
          min_expected = value<DstT>::nan(0);
          max_expected = value<DstT>::nan_max();
          return std::isnan(retrieved);
        }
      }
      // Now we are safe to assume any NaN value retrieved is an error
      if (std::isnan(retrieved)) {
        return false;
      }
      // Handle infinity cases explicitly
      if (sycl::isinf(min_expected)) {
        return retrieved == min_expected;
      }
      // Now handle the rounding of values
      // C++17 states it's implementation-defined which way (up or down) value
      // would round in case it couldn't be represented exactly in destination
      // type. For example,
      // for 32-bit float with 23 fraction bits the following is true
      //    const auto retrieved = static_cast<float>(16777217ul);
      //    assert((retrieved == 16777216.f) || (retrieved == 16777218.f));
      //
      // First of all, no need to do such check if we convert to a type with a
      // higher or same precision (or a higher or same range of exactly
      // representable values if we speak about conversion from a integral type
      // to a floating point type - an unsigned char would fit in the float for
      // example)
      if constexpr (value<SrcT>::digits2() <= value<DstT>::digits2()) {
        log::debug("- no precision loss possible, skipping ULP check...");
        return retrieved == min_expected;
      } else {
        log::debug("- precision loss possible, moving to ULP check...");
        // Note that static_cast<SrcT>(static_cast<DstT>(reference)) might
        // result in UB on SrcT range border values, so it's impossible to get a
        // rounding direction in a simple way, so let's make a less strict, but
        // much simplier check

        const auto lowest = value<DstT>::lowest();
        const auto max = value<DstT>::max();
        min_expected = value<DstT>::nextafter(min_expected, lowest);
        max_expected = value<DstT>::nextafter(max_expected, max);

        return ((retrieved >= min_expected) && (retrieved <= max_expected));
      }
    }
  }
};

} // namespace esimd_test::api::functional
