#include "vision.hpp"

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

namespace {
double seconds(LONGLONG ticks) {
    static const auto frequency = [] {
        LARGE_INTEGER value;
        QueryPerformanceFrequency(&value);
        return double(value.QuadPart);
    }();
    return ticks / frequency;
}

double now() {
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return seconds(counter.QuadPart);
}

void check(HRESULT result, const char* operation) {
    if (FAILED(result)) throw std::runtime_error(
        std::string(operation) + " failed (HRESULT " + std::to_string(result) + ")");
}

class Capture {
public:
    bool read(RECT region, Detector& detector, Detection& result);
    void save(const wchar_t* path);
    void reset();

private:
    void open(HMONITOR monitor);

    HMONITOR monitor_ = nullptr;
    DXGI_OUTPUT_DESC display_{};
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D> staging_;
    cv::Size size_;
    bool has_frame_ = false;
};

class FrameLease {
public:
    explicit FrameLease(IDXGIOutputDuplication* duplication) : duplication_(duplication) {}
    ~FrameLease() { duplication_->ReleaseFrame(); }
    FrameLease(const FrameLease&) = delete;
    FrameLease& operator=(const FrameLease&) = delete;
private:
    IDXGIOutputDuplication* duplication_;
};

class MappedTexture {
public:
    MappedTexture(ID3D11DeviceContext* context, ID3D11Texture2D* texture,
                  cv::Size size) : context_(context), texture_(texture) {
        D3D11_MAPPED_SUBRESOURCE mapped{};
        check(context->Map(texture, 0, D3D11_MAP_READ, 0, &mapped), "Map staging texture");
        image = cv::Mat(size, CV_8UC4, mapped.pData, mapped.RowPitch);
    }
    ~MappedTexture() { context_->Unmap(texture_, 0); }
    MappedTexture(const MappedTexture&) = delete;
    MappedTexture& operator=(const MappedTexture&) = delete;
    cv::Mat image;
private:
    ID3D11DeviceContext* context_;
    ID3D11Texture2D* texture_;
};

void Capture::reset() {
    has_frame_ = false;
    monitor_ = nullptr;
    staging_.Reset();
    duplication_.Reset();
    context_.Reset();
    device_.Reset();
}

void Capture::open(HMONITOR monitor) {
    reset();
    ComPtr<IDXGIFactory1> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "Create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT a = 0; factory->EnumAdapters1(a, adapter.ReleaseAndGetAddressOf()) == S_OK; ++a) {
        ComPtr<IDXGIOutput> output;
        for (UINT o = 0; adapter->EnumOutputs(o, output.ReleaseAndGetAddressOf()) == S_OK; ++o) {
            check(output->GetDesc(&display_), "Read monitor geometry");
            if (display_.Monitor != monitor) continue;
            if (display_.Rotation != DXGI_MODE_ROTATION_IDENTITY)
                throw std::invalid_argument("Native capture requires an unrotated display.");
            ComPtr<IDXGIOutput1> output1;
            check(output.As(&output1), "Get DXGI output");
            check(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
                &device_, nullptr, &context_), "Create D3D11 device");
            check(output1->DuplicateOutput(device_.Get(), &duplication_), "Duplicate desktop output");
            monitor_ = monitor;
            return;
        }
    }
    throw std::invalid_argument("No DXGI output found for the game window.");
}

bool Capture::read(RECT region, Detector& detector, Detection& result) {
    const auto started = now();
    const auto monitor = MonitorFromRect(&region, MONITOR_DEFAULTTONULL);
    if (!monitor) throw std::invalid_argument("Keep the game inside a display for native capture.");
    if (monitor_ != monitor) open(monitor);
    const auto screen = display_.DesktopCoordinates;
    if (region.left < screen.left || region.top < screen.top
        || region.right > screen.right || region.bottom > screen.bottom) {
        throw std::invalid_argument("Keep the whole game on one display for native capture.");
    }

    DXGI_OUTDUPL_FRAME_INFO info{};
    ComPtr<IDXGIResource> resource;
    const auto acquired = duplication_->AcquireNextFrame(0, &info, &resource);
    if (acquired == DXGI_ERROR_WAIT_TIMEOUT) return false;
    check(acquired, "Acquire desktop presentation");
    const FrameLease lease(duplication_.Get());
    if (!info.LastPresentTime.QuadPart && has_frame_) return false;

    ComPtr<ID3D11Texture2D> desktop;
    check(resource.As(&desktop), "Read desktop texture");
    const cv::Size size(region.right - region.left, region.bottom - region.top);
    if (size != size_ || !staging_) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = size.width;
        desc.Height = size.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        check(device_->CreateTexture2D(&desc, nullptr, staging_.ReleaseAndGetAddressOf()),
              "Allocate client-area staging texture");
        size_ = size;
    }
    const D3D11_BOX box{UINT(region.left - screen.left), UINT(region.top - screen.top), 0,
        UINT(region.right - screen.left), UINT(region.bottom - screen.top), 1};
    context_->CopySubresourceRegion(staging_.Get(), 0, 0, 0, 0, desktop.Get(), 0, &box);
    const MappedTexture mapped(context_.Get(), staging_.Get(), size_);
    const double captured = now();
    detector.inspect(mapped.image, result);
    has_frame_ = true;
    result.presented_at = info.LastPresentTime.QuadPart ? seconds(info.LastPresentTime.QuadPart) : captured;
    result.capture_seconds = captured - started;
    result.detection_seconds = now() - captured;
    return true;
}

void Capture::save(const wchar_t* path) {
    if (!has_frame_) throw std::runtime_error("No captured frame is available to save.");
    const MappedTexture mapped(context_.Get(), staging_.Get(), size_);
    std::vector<unsigned char> encoded;
    if (!cv::imencode(".png", mapped.image, encoded)) throw std::runtime_error("PNG encoding failed.");
    std::ofstream file(std::filesystem::path(path), std::ios::binary);
    file.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    if (!file) throw std::runtime_error("Could not write captured PNG.");
}

struct Session {
    explicit Session(cv::Rect roi) : detector(roi) {}
    Detector detector;
    Capture capture;
    std::string error;
};
} // namespace

#define EXPORT extern "C" __declspec(dllexport)

EXPORT int jumpbot_abi_version() { return 1; }

EXPORT Session* jumpbot_create(int x, int y, int width, int height) {
    try {
        static std::once_flag configure;
        std::call_once(configure, [] { cv::setNumThreads(1); });
        return new Session({x, y, width, height});
    }
    catch (...) { return nullptr; }
}

EXPORT void jumpbot_destroy(Session* session) { delete session; }

EXPORT const char* jumpbot_error(Session* session) { return session->error.c_str(); }

EXPORT int jumpbot_add_template(Session* session, const char* name, const void* data,
                               int width, int height) {
    try {
        session->detector.add_template(name, cv::Mat(height, width, CV_8UC1, const_cast<void*>(data)));
        return 1;
    } catch (const std::exception& e) { session->error = e.what(); return -1; }
}

EXPORT int jumpbot_poll(Session* session, int left, int top, int width, int height, Detection* result) {
    try {
        return session->capture.read({left, top, left + width, top + height}, session->detector, *result);
    } catch (const std::invalid_argument& e) {
        session->error = e.what();
        return -1;
    } catch (const std::exception& e) {
        session->error = e.what();
        session->capture.reset();
        return -2;
    }
}

EXPORT int jumpbot_inspect(Session* session, const void* data, int width, int height,
                           size_t stride, int channels, Detection* result) {
    try {
        const double started = now();
        session->detector.inspect(cv::Mat(height, width, CV_8UC(channels), const_cast<void*>(data), stride),
                                  *result);
        result->detection_seconds = now() - started;
        return 1;
    } catch (const std::exception& e) { session->error = e.what(); return -1; }
}

EXPORT int jumpbot_save(Session* session, const wchar_t* path) {
    try { session->capture.save(path); return 1; }
    catch (const std::exception& e) { session->error = e.what(); return -1; }
}
