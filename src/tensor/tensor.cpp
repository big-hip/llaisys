#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t expected = 1;                     // 后面维度的累积乘积，从 1 开始
    for (size_t i = _meta.shape.size(); i-- > 0;) {
        if (_meta.strides[i] != (ptrdiff_t)expected) return false;
        expected *= _meta.shape[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // order 必须是 0..ndim-1 的一个排列
    CHECK_ARGUMENT(order.size() == ndim(), "permute: order size must equal ndim");
    std::vector<bool> seen(ndim(), false);
    for (auto idx : order) {
        CHECK_ARGUMENT(idx < ndim(), "permute: dimension index out of range");
        CHECK_ARGUMENT(!seen[idx], "permute: duplicate dimension index");
        seen[idx] = true;
    }

    // 不搬数据：shape 和 strides 用同一个 order 重排，offset 不变
    std::vector<size_t> new_shape(ndim());
    std::vector<ptrdiff_t> new_strides(ndim());
    for (size_t i = 0; i < ndim(); i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(
        new Tensor(TensorMeta{_meta.dtype, new_shape, new_strides}, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1) 新 shape 的元素总数必须与原来一致
    size_t new_numel = 1;
    for (auto s : shape) {
        new_numel *= s;
    }
    CHECK_ARGUMENT(new_numel == numel(), "view: new shape has a different number of elements");

    // 2) 连续张量：直接换成行主序 strides，无需任何校验
    if (isContiguous()) {
        std::vector<ptrdiff_t> new_strides(shape.size());
        size_t stride = 1;
        for (size_t i = 1; i <= shape.size(); i++) {
            new_strides[shape.size() - i] = stride;
            stride *= shape[shape.size() - i];
        }
        return std::shared_ptr<Tensor>(
            new Tensor(TensorMeta{_meta.dtype, shape, new_strides}, _storage, _offset));
    }

    // 3) 非连续张量：把旧维度划分成若干组，每组合并成一个新维度。
    //    只有组内地址保持一致时 view 才安全：
    //    对组内每个旧维度 t，必须满足
    //        stride[t] == (组内 t 右侧尺寸乘积) × (组左端真实维度的 stride)
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t old_i = 0; // 下一个待消费的旧维度（组的左边界）
    for (size_t j = 0; j < shape.size(); j++) {
        if (shape[j] == 1) { // 长度为 1 的维度从不被步进，stride 无约束
            new_strides[j] = 1;
            continue;
        }
        size_t lo = old_i;
        size_t prod = 1;
        while (old_i < ndim() && prod < shape[j]) {
            prod *= _meta.shape[old_i];
            old_i++;
        }
        CHECK_ARGUMENT(prod == shape[j], "view: dimensions cannot be merged into the new shape");

        // 组 [lo, old_i) 合并成新维度 j，新 stride 取组左端真实维度的 stride
        size_t lo_real = lo;
        while (lo_real < old_i && _meta.shape[lo_real] == 1) {
            lo_real++;
        }
        new_strides[j] = _meta.strides[lo_real];

        // 校验组内地址一致性
        size_t flat_after = 1;
        for (size_t t = old_i; t-- > lo;) {
            if (_meta.shape[t] > 1) {
                CHECK_ARGUMENT(_meta.strides[t] == static_cast<ptrdiff_t>(flat_after) * new_strides[j],
                               "view: non-contiguous region cannot be viewed");
            }
            flat_after *= _meta.shape[t];
        }
    }
    CHECK_ARGUMENT(old_i == ndim(), "view: new shape leaves dimensions unmatched");

    return std::shared_ptr<Tensor>(
        new Tensor(TensorMeta{_meta.dtype, shape, new_strides}, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "slice: dim out of range");
    CHECK_ARGUMENT(start <= end && end <= _meta.shape[dim], "slice: invalid [start, end) range");

    // 只改被切维度的大小，strides 不变（切面不影响内存排布）
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    // 起点前移：_offset 单位是字节、stride 单位是元素，所以要乘 elementSize()
    size_t new_offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * elementSize();

    return std::shared_ptr<Tensor>(
        new Tensor(TensorMeta{_meta.dtype, new_shape, _meta.strides}, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 前置：切换到本张量所在设备，之后 runtime() 才对应正确的后端
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        this->numel() * this->elementSize(),
        LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    // 已经连续：共享 storage，无需拷贝
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    // 不连续：在相同设备上新建连续张量，按当前 strides 逐元素拷贝
    tensor_t result = create(_meta.shape, _meta.dtype, this->deviceType(), this->deviceId());
    const size_t n = numel();
    const size_t es = elementSize();
    const size_t nd = ndim();
    const auto &shape_ = _meta.shape;
    const auto &strides_ = _meta.strides;

    // 用一个"里程表"遍历所有多维下标：dst 按行主序线性递增，
    // src 用每个维度的下标 × stride 算出在内存里的元素偏移
    std::vector<size_t> index(nd, 0);
    for (size_t i = 0; i < n; i++) {
        size_t src_elem = 0;
        for (size_t d = 0; d < nd; d++) {
            src_elem += index[d] * static_cast<size_t>(strides_[d]);
        }
        std::memcpy(result->data() + i * es, data() + src_elem * es, es);

        // 下标进位（低维优先，和行主序一致）
        for (int d = static_cast<int>(nd) - 1; d >= 0; d--) {
            if (++index[d] < shape_[d]) {
                break;
            }
            index[d] = 0;
        }
    }
    return result;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // PyTorch 语义：连续张量 reshape 就是 view；非连续的先拷成连续再 view
    // （view 内部会校验元素总数一致）
    if (isContiguous()) {
        return view(shape);
    }
    auto contig = contiguous();
    return contig->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    // 目标设备就是当前设备：直接共享 storage
    if (device_type == this->deviceType() && (device < 0 || device == this->deviceId())) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    // 源不连续时先在同设备拷成连续，再搬过去（保证结果在目标设备上连续）
    if (!isContiguous()) {
        auto contig = contiguous();
        return contig->to(device_type, device);
    }

    int target_device = (device < 0) ? 0 : device;
    tensor_t result = create(_meta.shape, _meta.dtype, device_type, target_device);

    // 按源/目标设备决定拷贝方向
    llaisysMemcpyKind_t kind;
    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
    } else if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
    } else if (device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
    } else {
        kind = LLAISYS_MEMCPY_D2D;
    }

    // 在目标设备上下文里发起拷贝
    core::context().setDevice(device_type, target_device);
    core::context().runtime().api()->memcpy_sync(
        result->data(), this->data(), numel() * elementSize(), kind);
    return result;
}

} // namespace llaisys
