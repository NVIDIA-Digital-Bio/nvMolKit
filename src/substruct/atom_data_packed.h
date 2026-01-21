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

#ifndef NVMOLKIT_ATOM_DATA_PACKED_H
#define NVMOLKIT_ATOM_DATA_PACKED_H

#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define HD_CALLABLE __host__ __device__
#else
#define HD_CALLABLE
#endif

namespace nvMolKit {

/**
 * @brief Packed atom data for efficient GPU comparison via bitwise operations.
 *
 * This structure packs all atom properties into 128 bits (2x uint64_t) to enable
 * branchless mask-and-compare matching. All threads in a warp execute the same
 * instructions regardless of which fields are being compared.
 *
 * Bit layout:
 *
 * Lower 64 bits (lo):
 *   Byte 0 [bits  0-7 ]: atomicNum (uint8_t, valid range 0-118)
 *   Byte 1 [bits  8-15]: numExplicitHs (uint8_t)
 *   Byte 2 [bits 16-23]: explicitValence (uint8_t, 0xFF = unset)
 *   Byte 3 [bits 24-31]: implicitValence (uint8_t, 0xFF = unset)
 *   Byte 4 [bits 32-39]: formalCharge (int8_t, signed, stored as uint8_t)
 *   Byte 5 [bits 40-47]: chiralTag (uint8_t)
 *   Byte 6 [bits 48-55]: numRadicalElectrons (uint8_t)
 *   Byte 7 [bits 56-63]: hybridization (uint8_t)
 *
 * Upper 64 bits (hi):
 *   Byte 0 [bits  0-7 ]: minRingSize (uint8_t)
 *   Byte 1 [bits  8-15]: numRings (4 bits [8-11]) | ringBondCount (4 bits [12-15])
 *   Byte 2 [bits 16-23]: numImplicitHs (4 bits [16-19]) | numHeteroatomNeighbors (4 bits [20-23])
 *   Byte 3 [bits 24-31]: totalValence (uint8_t, explicit + implicit)
 *   Byte 4 [bits 32-39]: UNUSED. Reserved for future use.
 *   Byte 5 [bits 40-47]: isotope (uint8_t, 0 = natural abundance, throws if > 255)
 *   Byte 6 [bits 48-55]: degree (6 bits [48-53]) | isAromatic (bit 54) | isInRing (bit 55)
 *   Byte 7 [bits 56-63]: totalConnectivity (uint8_t, degree + total Hs for [X] queries)
 *
 */
struct AtomDataPacked {
  uint64_t lo = 0;
  uint64_t hi = 0;

  /// Sentinel value for unset valence fields
  static constexpr uint8_t kUnsetValence = std::numeric_limits<uint8_t>::max();

  // ============================================================================
  // Byte offset constants (within lo or hi)
  // ============================================================================

  /// @name Lower 64-bit field byte offsets
  /// @{
  static constexpr int kAtomicNumByte           = 0;
  static constexpr int kNumExplicitHsByte       = 1;
  static constexpr int kExplicitValenceByte     = 2;
  static constexpr int kImplicitValenceByte     = 3;
  static constexpr int kFormalChargeByte        = 4;
  static constexpr int kChiralTagByte           = 5;
  static constexpr int kNumRadicalElectronsByte = 6;
  static constexpr int kHybridizationByte       = 7;
  /// @}

  /// @name Upper 64-bit field byte offsets
  /// @{
  static constexpr int kMinRingSizeByte       = 0;
  static constexpr int kNumRingsRingBondsByte = 1;  ///< numRings (lo 4 bits) | ringBondCount (hi 4 bits)
  static constexpr int kImplicitHsHeterosByte = 2;  ///< numImplicitHs (lo 4 bits) | numHeteroatomNeighbors (hi 4 bits)
  static constexpr int kTotalValenceByte      = 3;
  static constexpr int kReservedByte          = 4;  ///< Reserved for future use
  static constexpr int kIsotopeByte           = 5;  ///< Isotope mass number (0 = natural abundance)
  static constexpr int kDegreeByte            = 6;  ///< Bits 0-5: degree, bit 6: isAromatic, bit 7: isInRing
  static constexpr int kTotalConnectivityByte = 7;  ///< Degree + total Hs for [X] queries
  /// @}

  /// @name Bit positions within the degree byte (byte 6 of hi)
  /// @{
  static constexpr int kDegreeBits    = 6;     ///< Number of bits for degree field (max 63)
  static constexpr int kDegreeMask    = 0x3F;  ///< Mask for 6-bit degree (bits 0-5)
  static constexpr int kIsAromaticBit = 6;     ///< Bit position within degree byte for isAromatic
  static constexpr int kIsInRingBit   = 7;     ///< Bit position within degree byte for isInRing
  /// @}

  /// @name Compacted 4-bit field masks and bit positions
  /// @{
  static constexpr int k4BitMask              = 0x0F;  ///< Mask for 4-bit fields (max value 15)
  static constexpr int kNumRingsBits          = 0;     ///< Low 4 bits of byte 1
  static constexpr int kRingBondCountBits     = 4;     ///< High 4 bits of byte 1
  static constexpr int kNumImplicitHsBits     = 0;     ///< Low 4 bits of byte 2
  static constexpr int kNumHeteroNeighborBits = 4;     ///< High 4 bits of byte 2
  static constexpr int kMax4BitValue          = 15;    ///< Maximum value for 4-bit fields
  /// @}

  // ============================================================================
  // Setters
  // ============================================================================

  HD_CALLABLE void setAtomicNum(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kAtomicNumByte * 8))) | (static_cast<uint64_t>(val) << (kAtomicNumByte * 8));
  }

  HD_CALLABLE void setNumExplicitHs(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kNumExplicitHsByte * 8))) | (static_cast<uint64_t>(val) << (kNumExplicitHsByte * 8));
  }

  HD_CALLABLE void setExplicitValence(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kExplicitValenceByte * 8))) | (static_cast<uint64_t>(val) << (kExplicitValenceByte * 8));
  }

  HD_CALLABLE void setImplicitValence(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kImplicitValenceByte * 8))) | (static_cast<uint64_t>(val) << (kImplicitValenceByte * 8));
  }

  HD_CALLABLE void setFormalCharge(int8_t val) {
    uint8_t uval = static_cast<uint8_t>(val);
    lo = (lo & ~(0xFFULL << (kFormalChargeByte * 8))) | (static_cast<uint64_t>(uval) << (kFormalChargeByte * 8));
  }

  HD_CALLABLE void setChiralTag(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kChiralTagByte * 8))) | (static_cast<uint64_t>(val) << (kChiralTagByte * 8));
  }

  HD_CALLABLE void setNumRadicalElectrons(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kNumRadicalElectronsByte * 8))) |
         (static_cast<uint64_t>(val) << (kNumRadicalElectronsByte * 8));
  }

  HD_CALLABLE void setHybridization(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kHybridizationByte * 8))) | (static_cast<uint64_t>(val) << (kHybridizationByte * 8));
  }

  HD_CALLABLE void setMinRingSize(uint8_t val) {
    hi = (hi & ~(0xFFULL << (kMinRingSizeByte * 8))) | (static_cast<uint64_t>(val) << (kMinRingSizeByte * 8));
  }

  HD_CALLABLE void setNumRings(uint8_t val) {
    constexpr uint64_t shift = kNumRingsRingBondsByte * 8 + kNumRingsBits;
    hi = (hi & ~(static_cast<uint64_t>(k4BitMask) << shift)) | (static_cast<uint64_t>(val & k4BitMask) << shift);
  }

  HD_CALLABLE void setRingBondCount(uint8_t val) {
    constexpr uint64_t shift = kNumRingsRingBondsByte * 8 + kRingBondCountBits;
    hi = (hi & ~(static_cast<uint64_t>(k4BitMask) << shift)) | (static_cast<uint64_t>(val & k4BitMask) << shift);
  }

  HD_CALLABLE void setNumImplicitHs(uint8_t val) {
    constexpr uint64_t shift = kImplicitHsHeterosByte * 8 + kNumImplicitHsBits;
    hi = (hi & ~(static_cast<uint64_t>(k4BitMask) << shift)) | (static_cast<uint64_t>(val & k4BitMask) << shift);
  }

  HD_CALLABLE void setNumHeteroatomNeighbors(uint8_t val) {
    constexpr uint64_t shift = kImplicitHsHeterosByte * 8 + kNumHeteroNeighborBits;
    hi = (hi & ~(static_cast<uint64_t>(k4BitMask) << shift)) | (static_cast<uint64_t>(val & k4BitMask) << shift);
  }

  HD_CALLABLE void setIsAromatic(bool val) {
    constexpr uint64_t bitPos = kDegreeByte * 8 + kIsAromaticBit;
    if (val) {
      hi |= (1ULL << bitPos);
    } else {
      hi &= ~(1ULL << bitPos);
    }
  }

  HD_CALLABLE void setTotalValence(uint8_t val) {
    hi = (hi & ~(0xFFULL << (kTotalValenceByte * 8))) | (static_cast<uint64_t>(val) << (kTotalValenceByte * 8));
  }

  HD_CALLABLE void setIsInRing(bool val) {
    constexpr uint64_t bitPos = kDegreeByte * 8 + kIsInRingBit;
    if (val) {
      hi |= (1ULL << bitPos);
    } else {
      hi &= ~(1ULL << bitPos);
    }
  }

  HD_CALLABLE void setIsotope(uint8_t val) {
    hi = (hi & ~(0xFFULL << (kIsotopeByte * 8))) | (static_cast<uint64_t>(val) << (kIsotopeByte * 8));
  }

  HD_CALLABLE void setDegree(uint8_t val) {
    constexpr uint64_t shift = kDegreeByte * 8;
    hi = (hi & ~(static_cast<uint64_t>(kDegreeMask) << shift)) | (static_cast<uint64_t>(val & kDegreeMask) << shift);
  }

  HD_CALLABLE void setTotalConnectivity(uint8_t val) {
    hi =
      (hi & ~(0xFFULL << (kTotalConnectivityByte * 8))) | (static_cast<uint64_t>(val) << (kTotalConnectivityByte * 8));
  }

  // ============================================================================
  // Getters
  // ============================================================================

  HD_CALLABLE uint8_t atomicNum() const { return static_cast<uint8_t>((lo >> (kAtomicNumByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t numExplicitHs() const { return static_cast<uint8_t>((lo >> (kNumExplicitHsByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t explicitValence() const {
    return static_cast<uint8_t>((lo >> (kExplicitValenceByte * 8)) & 0xFF);
  }

  HD_CALLABLE uint8_t implicitValence() const {
    return static_cast<uint8_t>((lo >> (kImplicitValenceByte * 8)) & 0xFF);
  }

  HD_CALLABLE int8_t formalCharge() const {
    return static_cast<int8_t>(static_cast<uint8_t>((lo >> (kFormalChargeByte * 8)) & 0xFF));
  }

  HD_CALLABLE uint8_t chiralTag() const { return static_cast<uint8_t>((lo >> (kChiralTagByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t numRadicalElectrons() const {
    return static_cast<uint8_t>((lo >> (kNumRadicalElectronsByte * 8)) & 0xFF);
  }

  HD_CALLABLE uint8_t hybridization() const { return static_cast<uint8_t>((lo >> (kHybridizationByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t minRingSize() const { return static_cast<uint8_t>((hi >> (kMinRingSizeByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t numRings() const {
    return static_cast<uint8_t>((hi >> (kNumRingsRingBondsByte * 8 + kNumRingsBits)) & k4BitMask);
  }

  HD_CALLABLE uint8_t ringBondCount() const {
    return static_cast<uint8_t>((hi >> (kNumRingsRingBondsByte * 8 + kRingBondCountBits)) & k4BitMask);
  }

  HD_CALLABLE uint8_t numImplicitHs() const {
    return static_cast<uint8_t>((hi >> (kImplicitHsHeterosByte * 8 + kNumImplicitHsBits)) & k4BitMask);
  }

  HD_CALLABLE uint8_t numHeteroatomNeighbors() const {
    return static_cast<uint8_t>((hi >> (kImplicitHsHeterosByte * 8 + kNumHeteroNeighborBits)) & k4BitMask);
  }

  HD_CALLABLE bool isAromatic() const {
    constexpr uint64_t bitPos = kDegreeByte * 8 + kIsAromaticBit;
    return (hi & (1ULL << bitPos)) != 0;
  }

  HD_CALLABLE uint8_t totalValence() const { return static_cast<uint8_t>((hi >> (kTotalValenceByte * 8)) & 0xFF); }

  HD_CALLABLE bool isInRing() const {
    constexpr uint64_t bitPos = kDegreeByte * 8 + kIsInRingBit;
    return (hi & (1ULL << bitPos)) != 0;
  }

  HD_CALLABLE uint8_t isotope() const { return static_cast<uint8_t>((hi >> (kIsotopeByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t degree() const { return static_cast<uint8_t>((hi >> (kDegreeByte * 8)) & kDegreeMask); }

  HD_CALLABLE uint8_t totalConnectivity() const {
    return static_cast<uint8_t>((hi >> (kTotalConnectivityByte * 8)) & 0xFF);
  }
};

static_assert(sizeof(AtomDataPacked) == 16, "AtomDataPacked must be exactly 16 bytes");

/**
 * @brief Precomputed mask and expected values for branchless atom matching.
 *
 * Generated on host from AtomQueryFlags during query molecule loading.
 * Enables single-instruction comparison: match = ((target & mask) == expected)
 *
 * For each atom property field:
 * - If the query flag is SET: corresponding mask byte = 0xFF, expected byte = query value
 * - If the query flag is NOT SET: corresponding mask byte = 0x00, expected byte = 0x00
 *
 * Special handling for aromaticity:
 * - AtomQueryIsAromatic: isAromatic mask = 0xFF, expected = 0x01 (must be true)
 * - AtomQueryIsAliphatic: isAromatic mask = 0xFF, expected = 0x00 (must be false)
 *
 * The match operation becomes:
 *   bool matches = ((target.lo & maskLo) == expectedLo) &&
 *                  ((target.hi & maskHi) == expectedHi);
 */
struct AtomQueryMask {
  uint64_t maskLo     = 0;
  uint64_t maskHi     = 0;
  uint64_t expectedLo = 0;
  uint64_t expectedHi = 0;
};

static_assert(sizeof(AtomQueryMask) == 32, "AtomQueryMask must be exactly 32 bytes");

/**
 * @brief Precomputed bond type counts per atom for efficient matching.
 *
 * Stores counts for single, double, triple, aromatic, and any bonds.
 * For target molecules: only single/double/triple/aromatic are populated.
 * For query molecules: 'any' holds SMARTS "~" bonds that match any type.
 *
 * Match semantics:
 * - Target must have >= of each specific bond type (single, double, triple, aromatic)
 * - Query "any" bonds can match any remaining bonds after specific types matched
 * - Therefore: target.total() >= query.total() must also be true.
 */
struct BondTypeCounts {
  uint8_t single   = 0;  ///< Single bonds (RDKit SINGLE=1)
  uint8_t double_  = 0;  ///< Double bonds (RDKit DOUBLE=2) (can't use double keyword)
  uint8_t triple   = 0;  ///< Triple bonds (RDKit TRIPLE=3)
  uint8_t aromatic = 0;  ///< Aromatic bonds (RDKit ONEANDAHALF=7 or AROMATIC=12)
  uint8_t any      = 0;  ///< "Any" bonds from SMARTS (~), only for queries

  HD_CALLABLE uint8_t total() const { return single + double_ + triple + aromatic + any; }

  /**
   * @brief Check if this atom can satisfy a query's bond requirements.
   *
   * For specific bond types (single, double, triple, aromatic), target must have >=.
   * For "any" bonds in query, they can be satisfied by any remaining target bonds,
   * so we just check that total target degree >= total query degree.
   *
   * @param query The query bond counts ('any' field holds SMARTS ~ bond count)
   * @return true if this target can match the query's bond requirements
   */
  HD_CALLABLE bool canMatchQuery(const BondTypeCounts& query) const {
    bool specificOk = (single >= query.single) && (double_ >= query.double_) && (triple >= query.triple) &&
                      (aromatic >= query.aromatic);
    // Short-circuit total() check if specific types are not met.
    return specificOk && (total() >= query.total());
  }
};

static_assert(sizeof(BondTypeCounts) == 5, "BondTypeCounts must be exactly 5 bytes");

/**
 * @brief Branchless atom matching using precomputed mask and expected values.
 *
 * All threads in a warp execute identical instructions regardless of which
 * fields are being compared, eliminating warp divergence.
 *
 * @param target The target atom to test
 * @param query The precomputed query mask with expected values
 * @return true if target matches query for all specified fields
 */
HD_CALLABLE inline bool atomMatchesPacked(const AtomDataPacked& target, const AtomQueryMask& query) {
  return ((target.lo & query.maskLo) == query.expectedLo) && ((target.hi & query.maskHi) == query.expectedHi);
}

/**
 * @brief Check if target atom has sufficient bonds to match query requirements.
 *
 * Handles "any" bonds from SMARTS: specific bond types must match exactly,
 * but "any" bonds can be satisfied by any remaining bonds in the target.
 *
 * @param target Bond counts for the target atom
 * @param query Bond counts for the query atom (other field = "any" bonds)
 * @return true if target can satisfy query's bond requirements
 */
HD_CALLABLE inline bool bondCountsMatchPacked(const BondTypeCounts& target, const BondTypeCounts& query) {
  return target.canMatchQuery(query);
}

}  // namespace nvMolKit

#undef HD_CALLABLE

#endif  // NVMOLKIT_ATOM_DATA_PACKED_H
