"""断点续传下载 HF 文件（纯标准库）。用法: python dl_hf.py <url> <dest>"""
import os
import sys
import time
import urllib.request


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    resp = urllib.request.urlopen(req, timeout=60)
    total = resp.headers.get("Content-Length")
    total = int(total) + have if total else 0
    if resp.status == 206:
        mode = "ab"
        print("续传自 %.1f MB" % (have / 1048576))
    else:
        mode = "wb"
        have = 0
        print("从头开始")

    done = have
    t0 = time.time()
    last = 0.0
    with open(part, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                sp = done / max(1e-9, time.time() - t0) / 1048576
                pct = 100.0 * done / total if total else 0
                print("  %5.1f%%  %7.1f / %.1f MB  %5.2f MB/s"
                      % (pct, done / 1048576, total / 1048576, sp), flush=True)

    if total and done < total:
        print("未完成，重跑本脚本可续传")
        sys.exit(1)
    if os.path.exists(dest):
        os.remove(dest)
    os.replace(part, dest)
    print("完成: %s  %.2f MB" % (dest, os.path.getsize(dest) / 1048576))


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
