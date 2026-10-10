#include "engine/PersistentGdnState.hpp"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>

using namespace splash::engine;

void testPersistentGdnExtentRoundtrip() {
  std::cout << "[testPersistentGdnExtentRoundtrip] Testing persistent GDN recurrent state serialization..." << std::endl;

  const uint64_t blockId = 0x1020304050607080ULL;
  const uint32_t layers = 48;
  const uint32_t boundaryTokens = 4096;
  const size_t convBytes = 16384;   // 16 KB conv state per layer
  const size_t recBytes = 131072;  // 128 KB recurrent state per layer

  PersistentGdnExtent original(blockId, layers, boundaryTokens, convBytes, recBytes);
  assert(original.verifyIntegrity());
  assert(original.kvBlockId() == blockId);
  assert(original.layers() == layers);
  assert(original.boundaryTokens() == boundaryTokens);

  // Populate synthetic state across layers
  for (uint32_t l = 0; l < layers; ++l) {
    float* conv = reinterpret_cast<float*>(original.layerConvData(l));
    float* rec = reinterpret_cast<float*>(original.layerRecurrentData(l));
    assert(conv != nullptr);
    assert(rec != nullptr);

    conv[0] = static_cast<float>(l * 1.5f);
    conv[100] = static_cast<float>(l * 2.5f);
    rec[0] = static_cast<float>(l * 10.0f);
    rec[500] = static_cast<float>(l * 20.0f);
  }

  std::string tempPath = "/tmp/test_gdn_extent.bin";

  // Measure write latency
  auto t0 = std::chrono::high_resolution_clock::now();
  bool saved = original.saveToFile(tempPath);
  auto t1 = std::chrono::high_resolution_clock::now();
  assert(saved);

  double write_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  size_t totalBytes = original.totalBytes();
  std::cout << "  Saved " << (totalBytes / (1024.0 * 1024.0)) << " MB in " << write_ms << " ms"
            << " (" << (totalBytes / (write_ms * 1e-3) / 1e9) << " GB/s)" << std::endl;

  // Measure load latency
  PersistentGdnExtent loaded;
  auto t2 = std::chrono::high_resolution_clock::now();
  bool loadedOk = loaded.loadFromFile(tempPath);
  auto t3 = std::chrono::high_resolution_clock::now();
  assert(loadedOk);
  assert(loaded.verifyIntegrity());

  double load_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  std::cout << "  Loaded " << (totalBytes / (1024.0 * 1024.0)) << " MB in " << load_ms << " ms"
            << " (" << (totalBytes / (load_ms * 1e-3) / 1e9) << " GB/s)" << std::endl;

  // Verify bit-exact fidelity across all layers
  for (uint32_t l = 0; l < layers; ++l) {
    const float* conv = reinterpret_cast<const float*>(loaded.layerConvData(l));
    const float* rec = reinterpret_cast<const float*>(loaded.layerRecurrentData(l));
    assert(conv[0] == static_cast<float>(l * 1.5f));
    assert(conv[100] == static_cast<float>(l * 2.5f));
    assert(rec[0] == static_cast<float>(l * 10.0f));
    assert(rec[500] == static_cast<float>(l * 20.0f));
  }

  std::filesystem::remove(tempPath);
  std::cout << "[testPersistentGdnExtentRoundtrip] PASS" << std::endl;
}

int main() {
  std::cout << "=== Running Persistent GDN State Tests ===" << std::endl;
  testPersistentGdnExtentRoundtrip();
  std::cout << "=== All Persistent GDN Tests Passed! ===" << std::endl;
  return 0;
}
