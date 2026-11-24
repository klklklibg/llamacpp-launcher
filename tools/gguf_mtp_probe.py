"""GGUF 头部解析：只读 metadata + tensor_info，判断模型是否带 MTP(nextn) 头。

不碰张量数据区，只扫描文件开头的元数据段；用 mmap 零拷贝避免逐条 seek。
"""
import mmap
import struct
import sys

# GGUF 元数据值类型 -> 定长字节数（STRING / ARRAY 单独处理）
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_T_STRING, _T_ARRAY = 8, 9


class _Cursor:
    """在 mmap 上的游标；跳过数据只移动指针，不拷贝。"""

    __slots__ = ("b", "p", "n")

    def __init__(self, buf):
        self.b = buf
        self.p = 0
        self.n = len(buf)

    def raw(self, k: int) -> bytes:
        if self.p + k > self.n:
            raise EOFError("truncated gguf")
        v = self.b[self.p:self.p + k]
        self.p += k
        return v

    def u32(self) -> int:
        return struct.unpack_from("<I", self.b, self._adv(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.b, self._adv(8))[0]

    def _adv(self, k: int) -> int:
        p = self.p
        if p + k > self.n:
            raise EOFError("truncated gguf")
        self.p = p + k
        return p

    def skip(self, k: int) -> None:
        if self.p + k > self.n:
            raise EOFError("truncated gguf")
        self.p += k

    def string(self, want: bool = False):
        n = self.u64()
        if self.p + n > self.n:
            raise EOFError("truncated gguf")
        s = None
        if want:
            s = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return s

    def skip_value(self, vtype: int) -> None:
        if vtype in _FIXED:
            self.skip(_FIXED[vtype])
        elif vtype == _T_STRING:
            self.string(want=False)
        elif vtype == _T_ARRAY:
            etype = self.u32()
            cnt = self.u64()
            if etype in _FIXED:
                self.skip(_FIXED[etype] * cnt)
            elif etype == _T_STRING:
                # 词表通常在这里：14 万条字符串，纯指针推进
                for _ in range(cnt):
                    self.skip(self.u64())
            else:
                for _ in range(cnt):
                    self.skip_value(etype)
        else:
            raise ValueError(f"unknown gguf value type {vtype}")


def probe(path: str) -> dict:
    """返回 {'ok':bool,'version':int,'tensors':n,'kv_has_nextn':bool,'tensor_has_nextn':bool}"""
    info = {"ok": False, "version": 0, "tensors": 0,
            "kv_has_nextn": False, "tensor_has_nextn": False}
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            return info            # 空文件 / 无法映射
        try:
            c = _Cursor(mm)
            if c.raw(4) != b"GGUF":
                return info
            version = c.u32()
            if version < 2:
                return info
            n_tensors = c.u64()
            n_kv = c.u64()
            info["ok"] = True
            info["version"] = version
            info["tensors"] = n_tensors

            for _ in range(n_kv):
                key = c.string(want=True) or ""
                vtype = c.u32()
                low = key.lower()
                if "nextn" in low or ".mtp" in low or low.endswith("mtp"):
                    info["kv_has_nextn"] = True
                c.skip_value(vtype)

            # tensor_info 紧跟 kv 之后，同样位于文件头部
            for _ in range(n_tensors):
                name = c.string(want=True) or ""
                n_dims = c.u32()
                c.skip(8 * n_dims)
                c.u32()   # ggml type
                c.u64()   # offset
                if "nextn" in name.lower():
                    info["tensor_has_nextn"] = True
        finally:
            try:
                mm.close()
            except BufferError:
                pass
    return info


def has_mtp(path: str) -> bool:
    try:
        i = probe(path)
    except (EOFError, OSError, ValueError):
        return False
    return bool(i["kv_has_nextn"] or i["tensor_has_nextn"])


if __name__ == "__main__":
    import time
    for p in sys.argv[1:]:
        t = time.perf_counter()
        i = probe(p)
        dt = (time.perf_counter() - t) * 1000
        print(f"{dt:7.1f} ms  mtp={i['kv_has_nextn'] or i['tensor_has_nextn']!s:5} "
              f"tensors={i['tensors']:5}  {p}")
