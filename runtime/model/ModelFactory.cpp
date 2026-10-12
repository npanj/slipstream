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
        } else {
          throw std::invalid_argument("unsupported target layout for load");
        }
      },
      result.descriptor.target);

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  if (std::filesystem::is_regular_file(root / "vision/model.bin")) {
    // The tower is shared; only the language-space projection width varies
    // with the text model (2560 for Flash-Next, 5120 for the 27B).
    ops::VisionLayout visionLayout;
    if (std::holds_alternative<Qwen4ExpWeights>(result.target)) {
      visionLayout.outputHiddenSize = 2560;
    }
    result.vision = loadQwenVisionWeights(backend, root / "vision", visionLayout);
    records.insert(records.end(), result.vision.files.begin(),
                   result.vision.files.end());
  }
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
