import ctypes, subprocess
from ctypes import wintypes as w

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)


class MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", w.DWORD), ("dwMemoryLoad", w.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class PMC(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def find_pid(name="llama-server.exe"):
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, errors="ignore")
    for line in out.stdout.splitlines():
        parts = line.strip().strip('"').split('","')
        if len(parts) >= 2 and parts[0].lower().startswith("llama-server"):
            return int(parts[1].replace(",", ""))
    return None


m = MEMSTATUS()
m.dwLength = ctypes.sizeof(m)
k32.GlobalMemoryStatusEx(ctypes.byref(m))
print(f"  SYSTEM  avail={m.ullAvailPhys / 2**30:.2f}GB  load={m.dwMemoryLoad}%")

pid = find_pid()
if pid:
    h = k32.OpenProcess(0x1000, False, pid)
    if h:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        ws = pmc.WorkingSetSize / 2**30
        pv = pmc.PrivateUsage / 2**30
        print(f"  PROC    WS={ws:.2f}GB  PRIV={pv:.2f}GB  MAPPED={ws - pv:.2f}GB")
        k32.CloseHandle(h)
else:
    print("  PROC    (none)")
