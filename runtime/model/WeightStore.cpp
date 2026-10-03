#include "WeightStore.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::model {

uint64_t checkedWeightMultiply(uint64_t left, uint64_t right,
                         std::string_view description) {
    if (left && right > std::numeric_limits<uint64_t>::max() / left) {
        throw WeightStoreError(std::string(description) + " overflows");
    }
    return left * right;
}

namespace {

[[nodiscard]] uint64_t checkedWeightAdd(uint64_t left, uint64_t right,
                                        std::string_view description) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw WeightStoreError(std::string(description) + " overflows");
    }
    return left + right;
}

[[nodiscard]] uint64_t q4Elements(uint32_t outputSize, uint32_t inputSize,
                                  uint32_t groupElements = kQ4GroupElements) {
    if (!outputSize || !inputSize || inputSize % groupElements) {
        throw WeightStoreError(
            "Q4 projection dimensions must be positive and input-aligned");
    }
    return checkedWeightMultiply(outputSize, inputSize, "Q4 element count");
}

[[nodiscard]] uint64_t q8PackedBytes(uint32_t outputSize, uint32_t inputSize) {
    uint64_t elements = q4Elements(outputSize, inputSize);
    return checkedWeightAdd(
        elements,
        checkedWeightMultiply(elements / 32, 2, "Q8 parameter byte count"),
        "Q8 packed byte count");
}

} // namespace

uint64_t q4PackedBytes(uint32_t outputSize, uint32_t inputSize,
                       uint32_t groupElements) {
    uint64_t elements = q4Elements(outputSize, inputSize, groupElements);
    // Half a byte per weight, plus a bf16 scale and bias per group.
    return checkedWeightAdd(
        elements / 2,
        checkedWeightMultiply(elements / groupElements, 2 * kBFloat16Bytes,
                              "Q4 parameter byte count"),
        "Q4 packed byte count");
}

void validateQ4Layout(uint32_t outputSize, uint32_t inputSize,
                      uint32_t storageN) {
    static_cast<void>(q4Elements(outputSize, inputSize));
    if (storageN != kQ4StorageN && storageN != kQ4ExpertStorageN) {
        throw WeightStoreError("unsupported Q4 StorageN");
    }
    if (outputSize % storageN) {
        throw WeightStoreError(
            "Q4 output dimension is incompatible with StorageN=" +
            std::to_string(storageN));
    }
}

namespace {

uint64_t alignPacked(uint64_t value) {
    return checkedWeightAdd(value, kWeightFileAlignment - 1,
                            "packed file alignment") &
           ~(kWeightFileAlignment - 1);
}

uint32_t loadLittleEndian32(const uint8_t *bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
        (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

std::string systemError(std::string_view operation,
                        const std::filesystem::path &path, int error) {
    return std::string(operation) + " " + path.string() + ": " +
        std::error_code(error, std::generic_category()).message();
}

class MappedRegion final {
public:
    static std::shared_ptr<MappedRegion> openReadOnly(
        const std::filesystem::path &path) {
        int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            throw WeightStoreError(systemError("unable to open", path, errno));
        }

        struct stat status {};
        if (fstat(descriptor, &status) != 0) {
            int error = errno;
            close(descriptor);
            throw WeightStoreError(systemError("unable to stat", path, error));
        }
        if (!S_ISREG(status.st_mode) || status.st_size <= 0) {
            close(descriptor);
            throw WeightStoreError("packed file is not a non-empty regular file: " +
                                   path.string());
        }
        uint64_t bytes = static_cast<uint64_t>(status.st_size);
        if (bytes > std::numeric_limits<size_t>::max()) {
            close(descriptor);
            throw WeightStoreError("packed file is too large to map: " +
                                   path.string());
        }

        // Metal can materialize MAP_PRIVATE file mappings as anonymous dirty
        // pages on GPU use. Keep immutable weights file-backed and reclaimable.
        void *address = mmap(nullptr, static_cast<size_t>(bytes), PROT_READ,
                             MAP_SHARED, descriptor, 0);
        int mapError = errno;
        close(descriptor);
        if (address == MAP_FAILED) {
            throw WeightStoreError(
                systemError("unable to mmap", path, mapError));
        }
        return std::shared_ptr<MappedRegion>(
            new MappedRegion(address, bytes));
    }

    ~MappedRegion() {
        if (address_) {
            munmap(address_, static_cast<size_t>(bytes_));
        }
    }

    MappedRegion(const MappedRegion &) = delete;
    MappedRegion &operator=(const MappedRegion &) = delete;

    [[nodiscard]] void *address() const noexcept { return address_; }
    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

    static int toMadvise(MemoryAdvice advice) noexcept {
        switch (advice) {
            case MemoryAdvice::Normal: return MADV_NORMAL;
            case MemoryAdvice::Random: return MADV_RANDOM;
            case MemoryAdvice::Sequential: return MADV_SEQUENTIAL;
            case MemoryAdvice::WillNeed: return MADV_WILLNEED;
            case MemoryAdvice::DontNeed: return MADV_DONTNEED;
        }
    }

    void advise(MemoryAdvice advice) const noexcept {
        if (address_ && bytes_) {
            madvise(address_, static_cast<size_t>(bytes_), toMadvise(advice));
        }
    }

    void adviseRange(uint64_t offset, uint64_t length, MemoryAdvice advice) const noexcept {
        if (address_ && offset < bytes_) {
            uint64_t actual = std::min(length, bytes_ - offset);
            madvise(static_cast<char *>(address_) + offset,
                    static_cast<size_t>(actual), toMadvise(advice));
        }
    }

    void prefetch(bool touch) const noexcept {
        if (address_ && bytes_) {
            madvise(address_, static_cast<size_t>(bytes_), MADV_WILLNEED);
            if (touch) {
                const auto *ptr = static_cast<const volatile char *>(address_);
                constexpr size_t kPage = 16 * 1024;
                for (size_t i = 0; i < bytes_; i += kPage) {
                    (void)ptr[i];
                }
            }
        }
    }

private:
    MappedRegion(void *address, uint64_t bytes)
        : address_(address), bytes_(bytes) {}

    void *address_ = nullptr;
    uint64_t bytes_ = 0;
};

} // namespace

struct WeightFile::Impl {
    metal::MetalBackend *backend = nullptr;
    std::shared_ptr<MappedRegion> mapping;
    metal::MetalBuffer base;
    WeightFileRecord record;
    uint64_t offset = 16;
    bool finished = false;
};

WeightFile::WeightFile(metal::MetalBackend &backend,
                       std::filesystem::path path,
                       std::string relativePath,
                       std::string_view expectedMagic,
                       uint32_t expectedLayer,
                       uint32_t expectedType)
    : impl_(std::make_unique<Impl>()) {
    if (expectedMagic.size() != 8) {
        throw WeightStoreError("packed file magic must contain eight bytes");
    }
    impl_->backend = &backend;
    impl_->mapping = MappedRegion::openReadOnly(path);
    if (impl_->mapping->bytes() < 16 ||
        impl_->mapping->bytes() % kWeightFileAlignment) {
        throw WeightStoreError(
            "packed file size is not 16 KiB-aligned: " + path.string());
    }
    const auto *header = static_cast<const uint8_t *>(
        impl_->mapping->address());
    uint32_t layer = loadLittleEndian32(header + 8);
    uint32_t type = loadLittleEndian32(header + 12);
    if (std::memcmp(header, expectedMagic.data(), 8) != 0 ||
        layer != expectedLayer || type != expectedType) {
        throw WeightStoreError("packed file header mismatch: " + path.string());
    }

    impl_->record = {
        std::move(relativePath), std::string(expectedMagic), layer, type,
        impl_->mapping->bytes(),
    };
    impl_->base = backend.wrapSharedMemory(
        impl_->mapping->address(), impl_->mapping->bytes(), impl_->mapping,
        impl_->record.relativePath);
}

WeightFile::~WeightFile() = default;

metal::MetalBuffer WeightFile::section(uint64_t bytes,
                                       std::string_view label) {
    if (impl_->finished) {
        throw WeightStoreError("cannot add a section after packed file finish");
    }
    if (!bytes) throw WeightStoreError("packed section must not be empty");
    uint64_t start = alignPacked(impl_->offset);
    uint64_t end = checkedWeightAdd(start, bytes, "packed section end");
    if (start % kWeightFileAlignment || end > impl_->mapping->bytes()) {
        throw WeightStoreError(
            "packed file is truncated at section " + std::string(label));
    }
    impl_->offset = end;
    return impl_->backend->view(impl_->base, start, bytes);
}

const uint8_t *WeightFile::mappedBase() const noexcept {
    return static_cast<const uint8_t *>(impl_->base.contents());
}

void WeightFile::finish() {
    if (impl_->finished) return;
    uint64_t consumed = alignPacked(impl_->offset);
    if (consumed != impl_->mapping->bytes()) {
        throw WeightStoreError(
            "packed file has unconsumed or missing bytes: " +
            impl_->record.relativePath);
    }
    impl_->finished = true;
}

const WeightFileRecord &WeightFile::record() const noexcept {
    return impl_->record;
}

uint64_t WeightFile::bytes() const noexcept {
    return impl_->mapping ? impl_->mapping->bytes() : 0;
}

void WeightFile::advise(MemoryAdvice advice) const noexcept {
    if (impl_->mapping) {
        impl_->mapping->advise(advice);
    }
}

void WeightFile::adviseRange(uint64_t offset, uint64_t bytes, MemoryAdvice advice) const noexcept {
    if (impl_->mapping) {
        impl_->mapping->adviseRange(offset, bytes, advice);
    }
}

void WeightFile::prefetch(bool touch) const noexcept {
    if (impl_->mapping) {
        impl_->mapping->prefetch(touch);
    }
}

ops::Q4Projection readQ4Projection(WeightFile &file,
                                   metal::MetalBackend &backend,
                                   uint32_t outputSize,
                                   uint32_t inputSize,
                                   std::string_view label,
                                   uint32_t storageN) {
    validateQ4Layout(outputSize, inputSize, storageN);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const uint64_t weightBytes = elements / 2;
    const uint64_t parameterBytes = elements / 32;
    metal::MetalBuffer packed =
        file.section(q4PackedBytes(outputSize, inputSize), label);
    return {
        backend.view(packed, 0, weightBytes),
        backend.view(packed, weightBytes, parameterBytes),
        backend.view(packed, weightBytes + parameterBytes, parameterBytes),
        outputSize,
        inputSize,
    };
}

ops::Q4Projection readQ4ProjectionComponents(WeightFile &file,
                                             uint32_t outputSize,
                                             uint32_t inputSize,
                                             std::string_view label,
                                             uint32_t groupElements) {
    const uint64_t elements = q4Elements(outputSize, inputSize, groupElements);
    const uint64_t parameters =
        checkedWeightMultiply(elements / groupElements, kBFloat16Bytes,
                              "Q4 component parameter bytes");
    const std::string prefix(label);
    return {
        file.section(elements / 2, prefix + "-weights"),
        file.section(parameters, prefix + "-scales"),
        file.section(parameters, prefix + "-biases"),
        outputSize,
        inputSize,
    };
}

ops::Q8Projection readQ8Projection(WeightFile &file,
                                   metal::MetalBackend &backend,
                                   uint32_t outputSize,
                                   uint32_t inputSize,
                                   std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const uint64_t parameterBytes = elements / 32;
    metal::MetalBuffer packed =
        file.section(q8PackedBytes(outputSize, inputSize), label);
    return {
        backend.view(packed, 0, elements),
        backend.view(packed, elements, parameterBytes),
        backend.view(packed, elements + parameterBytes, parameterBytes),
        outputSize,
        inputSize,
    };
}

ops::Q8Projection readQ8ProjectionComponents(WeightFile &file,
                                             uint32_t outputSize,
                                             uint32_t inputSize,
                                             std::string_view label) {
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::string prefix(label);
    return {
        file.section(elements, prefix + "-weights"),
        file.section(elements / 32, prefix + "-scales"),
        file.section(elements / 32, prefix + "-biases"),
        outputSize,
        inputSize,
    };
}

ops::ExpertQ4Projection
readExpertQ4Projection(WeightFile &file, uint32_t experts,
                       uint32_t outputSize, uint32_t inputSize,
                       std::string_view label, uint32_t storageN) {
    if (!experts)
        throw WeightStoreError("expert projection requires experts");
    validateQ4Layout(outputSize, inputSize, storageN);
    const uint64_t stride = q4PackedBytes(outputSize, inputSize);
    return {
        file.section(checkedWeightMultiply(experts, stride,
                                           "expert Q4 slab bytes"),
                     label),
        experts,
        outputSize,
        inputSize,
        stride,
    };
}

std::string weightManifestFingerprint(
    std::span<const WeightFileRecord> records) {
    std::vector<WeightFileRecord> sorted(records.begin(), records.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const WeightFileRecord &left,
                 const WeightFileRecord &right) {
                  return left.relativePath < right.relativePath;
              });
    std::ostringstream canonical;
    canonical << "splash-packed-manifest-v1\n";
    for (const WeightFileRecord &record : sorted) {
        canonical << record.relativePath << '\t' << record.declaredBytes
                  << '\t' << record.magic << '\t' << record.layer << '\t'
                  << record.type << '\n';
    }
    std::string value = canonical.str();
    if (value.size() > std::numeric_limits<CC_LONG>::max()) {
        throw WeightStoreError("manifest is too large to fingerprint");
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (!CC_SHA256(value.data(), static_cast<CC_LONG>(value.size()),
                   digest.data())) {
        throw WeightStoreError("unable to calculate manifest SHA-256");
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
}

} // namespace splash::model
