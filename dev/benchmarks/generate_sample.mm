#include "engine/Types.hpp"
#include "model/Runtime.hpp"
#include "ops/Q8PageStorage.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"

#import <Foundation/Foundation.h>

#include <fstream>
#include <algorithm>
#include <array>
#include <map>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

std::vector<uint32_t> pageRange(uint32_t first, uint32_t count) {
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index)
    result[index] = first + index;
  return result;
}

uint32_t parseCount(std::string_view text, std::string_view label) {
  uint32_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !value) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return value;
}

std::vector<uint32_t> parseTokens(std::string_view text) {
  std::vector<uint32_t> tokens;
  size_t start = 0;
  while (start < text.size()) {
    while (start < text.size() &&
           (text[start] == ',' || text[start] == ' ' || text[start] == '\n' ||
            text[start] == '\r'))
      ++start;
    if (start >= text.size())
      break;
    size_t end = start;
    while (end < text.size() && text[end] != ',' && text[end] != ' ' &&
           text[end] != '\n' && text[end] != '\r')
      ++end;
    uint32_t id = 0;
    auto res = std::from_chars(text.data() + start, text.data() + end, id);
    if (res.ec == std::errc{}) {
      tokens.push_back(id);
    }
    start = end;
  }
  return tokens;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) {
        std::cerr << "usage: generate-sample METALLIB MODEL_ROOT [MAX_TOKENS] "
                     "[TOKEN_ID_CSV or stdin]\n";
        return 2;
      }
      const char *metallibPath = argv[1];
      const char *modelRoot = argv[2];
      uint32_t maxTokens = 32;
      if (argc >= 4) {
        maxTokens = parseCount(argv[3], "max_tokens");
      }
      std::vector<uint32_t> prompt;
      std::vector<std::vector<uint32_t>> prompts;
      if (argc >= 5) {
        // Several prompts, separated by ';', run one after another on the
        // same engine, as a server would see them: caches stay warm.
        // "@path" reads the same ';'-separated prompts from a file, for
        // prompts too long for the command line.
        std::string fromFile;
        if (argv[4][0] == '@') {
          std::ifstream in(argv[4] + 1);
          if (!in)
            throw std::runtime_error(std::string("cannot read ") + (argv[4] + 1));
          fromFile.assign(std::istreambuf_iterator<char>(in), {});
          while (!fromFile.empty() && std::isspace(static_cast<unsigned char>(fromFile.back())))
            fromFile.pop_back();
        }
        std::string_view all = fromFile.empty() ? std::string_view(argv[4]) : fromFile;
        for (size_t start = 0; start <= all.size();) {
          const size_t end = std::min(all.find(';', start), all.size());
          prompts.push_back(parseTokens(all.substr(start, end - start)));
          start = end + 1;
        }
        prompt = prompts.front();
      } else {
        std::string input;
        std::string line;
        while (std::getline(std::cin, line)) {
          input += line + " ";
        }
        prompt = parseTokens(input);
        prompts.push_back(prompt);
      }
      for (const auto &each : prompts)
        if (each.size() > prompt.size())
          prompt = each;
      if (prompt.empty()) {
        std::cerr << "error: prompt tokens cannot be empty\n";
        return 1;
      }

      metal::MetalBackend backend(metallibPath);
      model::ModelPackage model = model::loadModelPackage(
          backend, std::filesystem::path(modelRoot));
      ops::ExecutionPlans operators(backend.capabilities());
      model::ModelMemoryPlan executorPlan =
          model::plannedRuntimeMemory(backend.capabilities(), model, operators);

      const uint32_t pagesPerLane =
          (prompt.size() + maxTokens +
           model::ExecutionLimits::targetVerifyRows) /
              kv::kPageTokens +
          4;
      const uint32_t pageCount =
          (pagesPerLane + model.targetKvLayout().sparseMappingBatchPages() - 1) /
          model.targetKvLayout().sparseMappingBatchPages() *
          model.targetKvLayout().sparseMappingBatchPages();

      const uint64_t governorLimit = std::max(
          backend.capabilities().recommendedMaxWorkingSetBytes,
          backend.memoryStats().allocatedBytes + 4ULL * 1024 * 1024 * 1024);
      MemoryGovernor governor(backend, governorLimit, 1);
      kv::Q8PageStorage pages(backend, governor.allocationAdmission(),
                              model.targetKvLayout(), pageCount);
      for (uint32_t page = 0; page < pageCount; ++page) {
        if (!pages.ensureResident(page))
          throw std::runtime_error("could not back the KV pages");
      }
      model::QwenStateStorage states(backend,
                                      governor.allocationAdmission(),
                                      model.stateLayout());
      model::RuntimeContext context{
          backend, governor.allocationAdmission(), model, pages, states, operators,
          ops::kMaximumImagePatches,
          executorPlan.pipelineReserveBytes,
          executorPlan.runtimeOverheadReserveBytes};
      model::Runtime executor(context);

      const uint32_t profileSteps = [] {
        const char *value = std::getenv("SPLASH_PROFILE_STEPS");
        return value ? static_cast<uint32_t>(std::atoi(value)) : 0u;
      }();
      std::map<std::string, std::pair<double, uint64_t>> profile;
      for (uint32_t promptIndex = 0; promptIndex < prompts.size(); ++promptIndex) {
        const std::vector<uint32_t> &prompt = prompts[promptIndex];
        const uint64_t laneId = 1 + promptIndex;
        const uint32_t slot = 0;
        std::vector<uint32_t> lanePages = pageRange(0, pagesPerLane);

        EngineRequest request;
        request.id = laneId;
        request.prompt.assign(prompt.begin(), prompt.end());
        request.maxNewTokens = maxTokens;
        // SPLASH_TEMPERATURE=t samples instead of taking the top token.
        if (const char *temperature = std::getenv("SPLASH_TEMPERATURE")) {
          request.sampling.temperature = static_cast<float>(std::atof(temperature));
          if (request.sampling.temperature > 0.0f) {
            // Qwen's recommended sampling: top-k 20, top-p 0.95.
            request.sampling.topK = 20;
            request.sampling.topP = 0.95f;
            request.cohort = BatchCohort::Sampling;
          }
        }
        executor.beginColdRequest(request.modelView(), slot);

        // Prefill phase
        const auto prefillStart = std::chrono::steady_clock::now();
        uint32_t offset = 0;
        std::vector<uint32_t> generatedTokens;
        while (offset < prompt.size()) {
          const uint32_t count = std::min<uint32_t>(
              model::ExecutionLimits::prefillTokenBudget,
              static_cast<uint32_t>(prompt.size()) - offset);
          BatchPlan plan{WorkKind::Prefill, request.cohort,
                         {{laneId, count, offset}}, DecodeStage::Regular};
          ModelBatchItem item{laneId, slot, offset, offset, count, lanePages};
          item.inputTokens =
              std::span<const uint32_t>(prompt).subspan(offset, count);
          // SPLASH_PROFILE_PREFILL=1 attributes the prefill's GPU time per kernel.
          const bool profilePrefill = std::getenv("SPLASH_PROFILE_PREFILL") != nullptr;
          if (profilePrefill)
            backend.setDispatchProfiling(true);
          auto results =
              executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
          if (profilePrefill) {
            backend.setDispatchProfiling(false);
            std::map<std::string, std::pair<double, uint64_t>> rows;
            for (const auto &timing : backend.takeDispatchProfile()) {
              rows[timing.pipelineName].first += timing.gpuSeconds;
              rows[timing.pipelineName].second += 1;
            }
            std::vector<std::pair<std::string, std::pair<double, uint64_t>>> sorted(rows.begin(), rows.end());
            std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
              return a.second.first > b.second.first;
            });
            for (const auto &row : sorted)
              std::cerr << "[prefill-profile] " << row.second.first * 1000 << " ms  x"
                        << row.second.second << "  " << row.first << "\n";
          }

          if (results.size() != 1 || results[0].consumedPromptTokens != count)
            throw std::runtime_error("prefill consumed wrong row count");
          // Scoring runs ask for the logits of every prompt position.
          if (const char *dump = std::getenv("SPLASH_DUMP_PREFILL_LOGITS"))
            executor.dumpPrefillLogits(count, dump);
          // Guesser training: inner state and top picks per prompt row.
          if (const char *dump = std::getenv("SPLASH_DUMP_PREFILL_FEATURES"))
            executor.dumpPrefillFeatures(count, dump);
          offset += count;

          for (uint32_t token : results[0].outputTokens) {
            generatedTokens.push_back(token);
          }
          if (results[0].finished) {
            std::cerr << "Prefill emitted stop token\n";
            break;
          }
        }
        const auto prefillFinish = std::chrono::steady_clock::now();
        const double prefillMs = std::chrono::duration<double, std::milli>(
                                     prefillFinish - prefillStart)
                                     .count();
        std::cerr << "Prefill completed in " << prefillMs << " ms. Prompt tokens: " << prompt.size() << "\n";

        std::vector<double> decodeStepMs;
        std::vector<uint32_t> decodeStepAccepted;

        uint64_t position = prompt.size();
        uint32_t totalGenerated = static_cast<uint32_t>(generatedTokens.size());
        while (totalGenerated < maxTokens) {
          BatchPlan plan{WorkKind::Decode, request.cohort,
                         {{laneId, 0, 0}}, DecodeStage::Regular};
          ModelBatchItem item{laneId, slot, position, 0, 0, lanePages};

          // SPLASH_PROFILE_STEPS=N times every GPU dispatch of N decode steps,
          // after the first eight, one dispatch per command so each is
          // attributed to its kernel. Profiled steps run slower than real ones.
          const bool profiling = profileSteps && decodeStepMs.size() >= 8 &&
                                 decodeStepMs.size() < 8 + profileSteps;
          if (profiling)
            backend.setDispatchProfiling(true);
          const auto stepStart = std::chrono::steady_clock::now();
          auto results =
              executor.decode(plan, std::span<const ModelBatchItem>(&item, 1));
          if (profiling) {
            backend.setDispatchProfiling(false);
            for (const auto &timing : backend.takeDispatchProfile()) {
              auto &entry = profile[timing.pipelineName];
              entry.first += timing.gpuSeconds;
              entry.second += 1;
            }
          }
          const auto stepFinish = std::chrono::steady_clock::now();
          const double stepMs = std::chrono::duration<double, std::milli>(
                                    stepFinish - stepStart)
                                    .count();
          decodeStepMs.push_back(stepMs);
          decodeStepAccepted.push_back(0);

          if (results.empty())
            break;
          const auto &res = results[0];
          if (res.outputTokens.empty())
            break;
          decodeStepAccepted.back() = res.acceptedDraftTokens;

          for (uint32_t token : res.outputTokens) {
            generatedTokens.push_back(token);
            ++totalGenerated;
            std::cerr << "Token: " << token << " (step: " << decodeStepMs.size() << ", total: " << totalGenerated << ")\n";
          }

          position += res.outputTokens.size() - res.outputTokensWithoutKv;

          if (res.finished) {
            std::cerr << "Finished by stop token\n";
            break;
          }
        }

        executor.end(laneId);

        // One line of JSON per prompt.
        std::cout << "{\"prefill_ms\": " << prefillMs
                  << ", \"prompt_tokens\": " << prompt.size()
                  << ", \"decode_steps\": " << decodeStepMs.size()
                  << ", \"generated_tokens\": [";
        for (size_t i = 0; i < generatedTokens.size(); ++i)
          std::cout << (i ? ", " : "") << generatedTokens[i];
        std::cout << "], \"step_ms\": [";
        for (size_t i = 0; i < decodeStepMs.size(); ++i)
          std::cout << (i ? ", " : "") << decodeStepMs[i];
        std::cout << "], \"step_accepted\": [";
        for (size_t i = 0; i < decodeStepAccepted.size(); ++i)
          std::cout << (i ? ", " : "") << decodeStepAccepted[i];
        std::cout << "]}" << std::endl;
      }

      if (profileSteps && !profile.empty()) {
        std::vector<std::pair<std::string, std::pair<double, uint64_t>>> rows(
            profile.begin(), profile.end());
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
          return a.second.first > b.second.first;
        });
        double total = 0;
        for (const auto &row : rows)
          total += row.second.first;
        std::cerr << "[profile] per decode step, GPU ms summed per kernel:\n";
        for (const auto &row : rows)
          std::cerr << "[profile] " << row.second.first * 1000 / profileSteps
                    << " ms  x" << row.second.second / profileSteps << "  "
                    << row.first << "\n";
        std::cerr << "[profile] total " << total * 1000 / profileSteps
                  << " ms per step\n";
      }
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "generate-sample error: " << error.what() << '\n';
      return 1;
    }
  }
}
