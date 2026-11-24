"""打印 GGUF 的 metadata KV（纯标准库）。用法: python gguf_kv.py <file> [key 过滤子串]"""
import mmap
import struct
import sys

_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class C:
    def __init__(self, b):
        self.b, self.p, self.n = b, 0, len(b)

    def raw(self, k):
        if self.p + k > self.n:
            raise EOFError
        v = self.b[self.p:self.p + k]
        self.p += k
        return v

    def u32(self):
        return struct.unpack_from("<I", self.b, self._a(4))[0]

    def u64(self):
        return struct.unpack_from("<Q", self.b, self._a(8))[0]

    def _a(self, k):
        p = self.p
        if p + k > self.n:
            raise EOFError
        self.p = p + k
        return p

    def s(self):
        n = self.u64()
        if self.p + n > self.n:
            raise EOFError
        v = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return v

    def val(self, t):
        if t in _FMT:
            sz = _SZ[t]
            return struct.unpack_from(_FMT[t], self.b, self._a(sz))[0]
        if t == 8:
            return self.s()
        if t == 9:
            et, cnt = self.u32(), self.u64()
            if cnt > 64:                      # 大数组只看前几个
                head = [self.val(et) for _ in range(4)]
                for _ in range(cnt - 4):
                    self.val(et)
                return head + ["...共 %d 项" % cnt]
            return [self.val(et) for _ in range(cnt)]
        raise ValueError("type %d" % t)


def read_kv(path, filt=None):
    out = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            c = C(mm)
            if c.raw(4) != b"GGUF":
                return out
            ver = c.u32()
            n_tensors, n_kv = c.u64(), c.u64()
            out["__version__"] = ver
            out["__tensors__"] = n_tensors
            for _ in range(n_kv):
                k = c.s()
                t = c.u32()
                v = c.val(t)
                if filt is None or filt in k:
                    out[k] = v
        finally:
            try:
                mm.close()
            except BufferError:
                pass
    return out


if __name__ == "__main__":
    p = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 else None
    for k, v in read_kv(p, flt).items():
        vs = str(v)
        print("%-40s %s" % (k, vs[:100]))
