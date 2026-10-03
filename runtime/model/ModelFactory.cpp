#include "ModelFactory.hpp"

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace splash::model {

void requireCompatibleModelPackage(const ModelPackage &package) {
  if (!package.descriptor.valid() ||
      !std::visit(
          [&](const auto &target) {
            return package.descriptor.target == TargetLayout{target.layout};
          },
          package.target)) {
    throw std::invalid_argument("model weights do not match the descriptor");
  }
}

namespace {

ModelPackage loadPackage(metal::MetalBackend &backend,
                         const std::filesystem::path &root,
                         ModelDescriptor descriptor) {
  ModelPackage result;
  result.descriptor = std::move(descriptor);
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        using L = std::remove_cvref_t<decltype(layout)>;
        if constexpr (std::is_same_v<L, Qwen4ExpLayout>) {
          return loadQwen4ExpWeights(backend, root / "target", layout);
        } else if constexpr (std::is_same_v<L, Qwen3_8Layout>) {
          return loadQwen3_8Weights(backend, root / "target", layout);
        } else if constexpr (std::is_same_v<L, Qwen3_8Q8Layout>) {
          return loadQwen3_8Q8Weights(backend, root / "target", layout);
        }
      },
      result.descriptor.target);

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModelPackage(result);
  return result;
}

} // namespace

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root) {
  return loadPackage(backend, root, inspectModelPackage(root));
}

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor) {
  return loadPackage(backend, root, descriptor);
}

} // namespace splash::model
