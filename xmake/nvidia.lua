target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        -- 链接进 .so 需要 PIC：-Xcompiler 把 -fPIC 传给宿主编译器
        add_cuflags("-Xcompiler -fPIC", {force = true})
    end

    -- 用 nvcc 编译 .cu 文件
    add_rules("cuda")
    -- 关闭 RDC：让每个 .cu 对象自带 fatbin，这样 g++ 链接 .so 时符号自洽
    set_values("cuda.rdc", false)

    add_files("../src/device/nvidia/*.cu")

    -- 链接 CUDA 运行时
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler -fPIC", {force = true})
    end

    add_rules("cuda")
    set_values("cuda.rdc", false)

    -- 每个算子的 nvidia/ 子目录（与 CPU 的 cpu/ 子目录对应）
    add_files("../src/ops/*/nvidia/*.cu")

    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")

    on_install(function (target) end)
target_end()
