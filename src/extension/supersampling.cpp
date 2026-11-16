#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

#include "../common/headless.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    log_level_verbose();
    // Consume --headless/--cli (plus the generic --frames/--output-dir) so the
    // uniform gallery CLI is accepted; this example never opens a window.
    auto headless = luisa::gallery::headless::parse(argc, argv);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--headless]. <backend>: dx", argv[0]);
        return 1;
    }
    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Supersampling example: not yet implemented.");
    LUISA_INFO("supersampling: success, nothing to run yet (headless={}).", headless.enabled());
    return 0;
}
