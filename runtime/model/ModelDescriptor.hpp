#pragma once

#include "Model.hpp"
#include "models/qwen4exp/Qwen4Exp.hpp"
#include "models/qwen38/Qwen3_8.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

namespace splash::model {

using TargetLayout =
    std::variant<Qwen4ExpLayout, Qwen3_8Layout, Qwen3_8Q8Layout>;

// Package metadata validated before weight buffers are loaded. The engine
// consumes capabilities; model loading consumes the concrete layouts.
struct ModelDescriptor final {
  std::string name;
  TargetLayout target;
  ModelCapabilities capabilities;
  kv::Q8Layout targetKvLayout;
  CompositeStateLayout stateLayout;
  // Exact bytes parsed during package inspection, including artifact digests.
  // Synthetic descriptors retain zero; this is separate from layout identity.
  std::array<uint8_t, 32> packageManifestSha256{};

  [[nodiscard]] bool valid() const noexcept;
};

// Derives the capabilities and cache layouts the engine consumes from the
// concrete target layout.
[[nodiscard]] ModelDescriptor makeModelDescriptor(std::string name,
                                                  TargetLayout target);
[[nodiscard]] ModelDescriptor
inspectModelPackage(const std::filesystem::path &root);

} // namespace splash::model
