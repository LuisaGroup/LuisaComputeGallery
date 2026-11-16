#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/backends/ext/dx_cuda_interop.h>

#include "../common/headless.h"

using namespace luisa;
using namespace luisa::compute;

// Usage: example_cuda_dx_interop [--headless]. Always runs on dx + cuda and
// never opens a window.
int main(int argc, char *argv[]) {
    // Consume --headless/--cli (plus the generic --frames/--output-dir) so the
    // uniform gallery CLI parses; the work below is the same in both modes.
    auto headless = luisa::gallery::headless::parse(argc, argv);
    Context context{argv[0]};
    Device dx_device = context.create_device("dx");
    auto interop_ext = dx_device.extension<DxCudaInterop>();
    DeviceConfig cuda_device_config{
        .device_index = static_cast<size_t>(interop_ext->cuda_device_index())};
    Device cuda_device = context.create_device("cuda");
    Stream cuda_stream = cuda_device.create_stream();
    Stream dx_stream = dx_device.create_stream();

    auto interop_event = interop_ext->create_timeline_event();

    auto interop_buffer = interop_ext->create_buffer<uint>(1);
    uint64_t cuda_ptr;
    uint64_t cuda_handle;
    interop_ext->cuda_buffer(interop_buffer.handle(), &cuda_ptr, &cuda_handle);
    auto cuda_buffer = cuda_device.import_external_buffer<uint>(reinterpret_cast<void *>(cuda_ptr), 1);
    uint input = 114514;
    uint output{};
    dx_stream << interop_buffer.copy_from(luisa::span{&input, 1}) << interop_event.dx_signal(1);
    cuda_stream << interop_event.cuda_wait(1) << cuda_buffer.copy_to(luisa::span{&output, 1}) << synchronize();
    LUISA_INFO("Result: {}", output);
    interop_ext->unmap(reinterpret_cast<void *>(cuda_ptr), reinterpret_cast<void *>(cuda_handle));

    // Always remember to synchronize event it-self!
    interop_event.dx_event.synchronize(1);

    auto ok = output == input;
    if (!ok) { LUISA_WARNING("dx -> cuda interop mismatch: expected {}, got {}.", input, output); }
    LUISA_INFO("cuda_dx_interop: {} (result={}, headless={}).",
               ok ? "PASSED" : "FAILED", output, headless.enabled());
    return ok ? 0 : 1;
}