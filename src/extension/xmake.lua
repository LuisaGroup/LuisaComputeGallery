-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- condition is a DISJUNCTION in both build systems (dx OR vk OR cuda OR metal backend enabled) which the AND-only 'requires' list cannot express: keep requires empty and gate on (dx||vk||cuda||metal). xmake: if lc_dx_backend or lc_vk_backend or lc_cuda_backend or lc_metal_backend
gallery_example("example_dstorage", {sources = {"dstorage.cpp"}, requires_any = {"dx", "vk", "cuda", "metal"}})
-- creates its own DX/DirectML device; argv[1] only tags the run
gallery_example("example_dml", {sources = {"dml.cpp"}, requires = {"dx"}})
-- headless path prints "not yet implemented" and exits
gallery_example("example_supersampling", {sources = {"supersampling.cpp"}, requires = {"dx", "gui"}, gui_dep = true})
-- xmake: only when mode != debug and lc_enable_clangcxx, and it includes extension/clangcxx_compiler; cmake: only when LUISA_COMPUTE_ENABLE_CLANG_CXX, outside the GUI guard, and it does NOT build clangcxx_compiler ('xmake-only')
gallery_example("example_path_tracing_clangcxx", {sources = {"path_tracing_clangcxx.cpp"}, deps = {"lc-clangcxx"}, requires = {"gui", "clangcxx"}, gui_dep = true})
gallery_example("example_cuda_lcub", {sources = {"cuda_lcub.cpp"}, deps = {"lc-compute-cuda-ext-lcub"}, requires = {"cuda", "cuda_ext_lcub"}})
-- example_dx_supersampling: no xmake target (cmake only; see scripts/gallery_targets.json)
-- example_fsr3: no xmake target (cmake only; see scripts/gallery_targets.json)
-- xmake only (the cmake file says 'clangcxx_compiler is xmake-only; no CMake target available'). xmake sets enable_exception=true, forces an 8 MiB stack (/STACK:8388608 or -Wl,--stack,8388608) and, after_build on windows, copies clangcxx_compiler.exe + dxcompiler.dll + dxil.dll + luisa-clangcxx.dll + luisa-core.dll + luisa-runtime.dll (+ backend dlls) into build/clangcxx_compiler. It also needs the non-source assets extension/clangcxx_compiler/template.txt and simplecpp.{h,cpp}. Unlike example_proj targets it does NOT depend on lc-dsl
gallery_example("clangcxx_compiler", {sources = {"clangcxx_compiler/*.cpp"}, deps = {"lc-clangcxx", "reproc"}, requires = {"clangcxx"}})
