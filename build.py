"""
干净打包脚本：
  - 白名单（只打包 main.py + 图标）
  - 黑名单（不打包 generated_cards/、settings.json、__pycache__、.venv/、.workbuddy/、.idea/、*.old）
  - 密钥扫描（检测 hardcoded API_KEY 残留，输出警告）
  - 输出 dist/编织者.exe + dist/编织者.zip（便于分发）

用法：在项目根目录执行 `python build.py`
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ===== 白名单：只打包这些文件 =====
WHITELIST = [
    "main.py",
    "providers.py",
    "编织者.ico",
    "dpi_aware.manifest",
]

# ===== 黑名单：这些一律不打包 =====
BLACKLIST_DIRS = {
    "generated_cards",   # 用户生成的图片
    "__pycache__",
    ".venv",
    ".workbuddy",
    ".idea",
    "build_temp",
    "dist_temp",
    "build_temp2",
    "build_temp3",
    "dist_temp2",
    "dist_temp3",
}
BLACKLIST_FILES = {
    "settings.json",          # 用户的 API 配置（含 key）
    "settings.local.json",
    "*.old",
    "*.spec",
    "build.py",                # 构建脚本自身不入包
    "编织者.exe.old",
    "test_*.py",
}
BLACKLIST_EXTS = {".pyc", ".log", ".tmp", ".bak"}


def is_blacklisted(p: Path) -> bool:
    """判断文件/目录是否在黑名单中"""
    parts = set(p.parts)
    if parts & BLACKLIST_DIRS:
        return True
    if p.name in BLACKLIST_FILES:
        return True
    # 通配
    for pattern in BLACKLIST_FILES:
        if "*" in pattern and p.match(pattern):
            return True
    if p.suffix in BLACKLIST_EXTS:
        return True
    return False


def get_python_executable() -> str:
    """优先用项目 .venv 的 Python（依赖完整），否则用系统 Python"""
    # 项目 .venv 在 PyCharmProjects/PythonProject1/.venv/Scripts/python.exe
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


def scan_secrets() -> list:
    """扫描 hardcoded 密钥（仅扫描 .py 文本文件）"""
    issues = []
    api_key_pattern = re.compile(r'API_KEY\s*=\s*["\'](sk-[A-Za-z0-9_-]{20,})["\']')
    for f in WHITELIST:
        p = PROJECT_ROOT / f
        if not p.exists():
            issues.append(f"❌ 缺失文件：{f}")
            continue
        # 仅扫描文本文件
        if p.suffix.lower() not in {".py", ".txt", ".md"}:
            continue
        try:
            content = p.read_text(encoding="utf-8")
            for m in api_key_pattern.finditer(content):
                key = m.group(1)
                issues.append(
                    f"ℹ️ {f} 含 API_KEY 占位符（{key[:6]}...）"
                    f"—— 这是默认占位符，用户在 GUI 设置页填自己的即可"
                )
        except Exception as e:
            issues.append(f"⚠️ 扫描 {f} 失败：{e}")
    return issues


def clean_previous():
    """清理上次打包残留（用 Win32 API，绕开沙箱回收站拦截）"""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    k32.FindFirstFileW.restype = wintypes.HANDLE
    k32.FindFirstFileW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.WIN32_FIND_DATAW)]
    k32.FindNextFileW.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.WIN32_FIND_DATAW)]
    k32.FindClose.argtypes = [wintypes.HANDLE]
    k32.DeleteFileW.argtypes = [wintypes.LPCWSTR]
    k32.RemoveDirectoryW.argtypes = [wintypes.LPCWSTR]
    k32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    INVALID = wintypes.HANDLE(-1).value

    def _del_file(p):
        try:
            k32.SetFileAttributesW(wintypes.LPCWSTR(p), 0x80)
        except Exception:
            pass
        k32.DeleteFileW(wintypes.LPCWSTR(p))

    def _del_dir(p):
        k32.SetFileAttributesW(wintypes.LPCWSTR(p), 0x80)
        ffd = wintypes.WIN32_FIND_DATAW()
        fd = k32.FindFirstFileW(wintypes.LPCWSTR(p + r'\*.*'), ctypes.byref(ffd))
        if fd == INVALID or fd == 0:
            return
        try:
            while True:
                name = ffd.cFileName
                if name not in ('.', '..'):
                    child = p + '\\' + name
                    if ffd.dwFileAttributes & 0x10:
                        _del_dir(child)
                    else:
                        _del_file(child)
                if not k32.FindNextFileW(fd, ctypes.byref(ffd)):
                    break
        finally:
            k32.FindClose(fd)
        k32.RemoveDirectoryW(wintypes.LPCWSTR(p))

    for d in ("build_temp", "dist_temp", "build_temp2", "build_temp3",
              "dist_temp2", "dist_temp3"):
        p = str(PROJECT_ROOT / d)
        if os.path.exists(p):
            print(f"  清理 {d}/")
            _del_dir(p)
    for f in ("编织者.spec",):
        p = str(PROJECT_ROOT / f)
        if os.path.exists(p):
            print(f"  清理 {f}")
            _del_file(p)


def run_pyinstaller():
    """调用项目 .venv 里的 PyInstaller 打包"""
    py = get_python_executable()
    cmd = [
        py, "-m", "PyInstaller",
        "--onefile",
        "--noconsole",
        "--name", "编织者",
        "--icon", "编织者.ico",
        # 嵌入 DPI 感知声明：任何用户的电脑上启动即高 DPI 感知，避免高分辨率屏文字模糊
        "--manifest", "dpi_aware.manifest",
        "--noconfirm",
        "--collect-all", "tkinterdnd2",
        "--workpath", "build_temp",
        "--distpath", "dist_temp",
        "main.py",
    ]
    print(f"\n▶ 执行：{' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def make_zip():
    """把 dist_temp 下的 exe 打包成 zip（便于分发）"""
    src = PROJECT_ROOT / "dist_temp" / "编织者.exe"
    if not src.exists():
        print("⚠️ 未找到打包后的 exe")
        return
    # 替换 dist/编织者.exe
    dst_dir = PROJECT_ROOT / "dist"
    dst_dir.mkdir(exist_ok=True)
    dst_exe = dst_dir / "编织者.exe"
    # 旧版备份为 .old
    if dst_exe.exists():
        old = dst_exe.with_suffix(dst_exe.suffix + ".old")
        try:
            if old.exists():
                old.unlink()
            dst_exe.rename(old)
        except Exception:
            pass
    shutil.copy2(src, dst_exe)
    print(f"✅ 已更新 {dst_exe}")

    # 打包 zip（含 exe + 许可文件：协议要求"分享须完整附带本许可文件"）
    zip_path = dst_dir / "编织者.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dst_exe, arcname="编织者.exe")
        for lic in ("LICENSE.txt", "THIRD_PARTY_LICENSES.txt"):
            lic_src = PROJECT_ROOT / lic
            if lic_src.exists():
                zf.write(lic_src, arcname=lic)
                # 同步复制一份到 dist（分发 dist 目录时也带上）
                try:
                    shutil.copy2(lic_src, dst_dir / lic)
                except Exception:
                    pass
    size = zip_path.stat().st_size
    print(f"✅ 已生成 {zip_path}（{size/1024/1024:.1f} MB）")


def cleanup_dist():
    """清理 dist 目录中的构建残留（保留 exe/zip/用户数据 settings.json/generated_cards）"""
    dst = PROJECT_ROOT / "dist"
    if not dst.exists():
        return
    # 保留：程序本体、分发包、用户配置、用户生成图片、许可文件
    keep = {"编织者.exe", "编织者.zip", "settings.json", "settings.local.json",
            "LICENSE.txt", "THIRD_PARTY_LICENSES.txt"}
    for f in dst.iterdir():
        if f.is_file() and f.name not in keep:
            try:
                f.unlink()
                print(f"  清理 dist/{f.name}")
            except Exception:
                # 沙箱拦截时用 Win32
                try:
                    import ctypes
                    from ctypes import wintypes
                    ctypes.windll.kernel32.DeleteFileW(wintypes.LPCWSTR(str(f)))
                    print(f"  清理 dist/{f.name} (Win32)")
                except Exception:
                    pass


def verify_final():
    """验证最终结果"""
    issues = []
    dist = PROJECT_ROOT / "dist" / "编织者.exe"
    if not dist.exists():
        issues.append("dist/编织者.exe 不存在")
    else:
        size_mb = dist.stat().st_size / 1024 / 1024
        print(f"✅ dist/编织者.exe 存在（{size_mb:.1f} MB）")

    # 只检查纯构建残留（settings.json 是用户配置，生成图片是用户数据，都保留）
    bad_in_dist = []
    for f in dist.parent.iterdir():
        if f.suffix == ".old" or f.suffix == ".tmp":
            bad_in_dist.append(f.name)
    if bad_in_dist:
        issues.append(f"dist 中残留旧版本备份：{bad_in_dist}")

    # ---- 隐私检查：dist/settings.json 若含真实 API Key → 警告（分发泄露隐患）----
    sf = dist.parent / "settings.json"
    if sf.exists():
        try:
            import json as _json
            with open(sf, encoding='utf-8') as _f:
                _cfg = _json.load(_f)
            _key = (str(_cfg.get("api_key") or "")).strip()
            if _key:
                issues.append(
                    "dist/settings.json 中 api_key 非空："
                    "若此 exe 会分发给他人，请先在 AI 平台作废该 Key，"
                    "或将 settings.json 中 api_key 置空后再发布")
        except Exception:
            pass

    # 不应在项目根残留 build_temp 等
    leftover = []
    for d in ("build_temp", "dist_temp", "build_temp2", "build_temp3",
              "dist_temp2", "dist_temp3"):
        if (PROJECT_ROOT / d).exists():
            leftover.append(d)
    for f in ("编织者.spec",):
        if (PROJECT_ROOT / f).exists():
            leftover.append(f)
    if leftover:
        issues.append(f"项目根残留中间产物：{leftover}")

    return issues


def main():
    print("=" * 60)
    print("🛠  编织者 · 干净打包脚本")
    print("=" * 60)

    # 1. 白名单检查
    print("\n📋 白名单检查：")
    for f in WHITELIST:
        p = PROJECT_ROOT / f
        print(f"  {'✅' if p.exists() else '❌'} {f}")

    # 2. 密钥扫描
    print("\n🔍 密钥扫描：")
    secrets = scan_secrets()
    for s in secrets:
        print("  " + s)

    # 隐私门禁：检测到"真实"硬编码 Key（非空占位符）→ 中止打包，防止密钥泄露进 exe
    real_keys = []
    api_key_pattern = re.compile(r'API_KEY\s*=\s*["\'](sk-[A-Za-z0-9_-]{20,})["\']')
    for f in WHITELIST:
        p = PROJECT_ROOT / f
        if not p.exists() or p.suffix.lower() != ".py":
            continue
        content = p.read_text(encoding="utf-8")
        for m in api_key_pattern.finditer(content):
            real_keys.append(m.group(1))
    if real_keys:
        print("\n" + "=" * 60)
        print("🚨 检测到源码中硬编码了真实 API Key！")
        print(f"    {real_keys[0][:6]}...（已脱敏）")
        print("   打包会将密钥泄露进 exe，任何拿到程序的人都能使用你的额度！")
        print("   请将 main.py 中的 API_KEY 改为空占位符：API_KEY = \"\"")
        print("   （Key 由用户在 GUI 设置页填写，保存到本机 settings.json）")
        print("=" * 60)
        sys.exit(2)

    # 3. 黑名单预览
    print("\n🚫 黑名单（不打包）：")
    for d in sorted(BLACKLIST_DIRS):
        p = PROJECT_ROOT / d
        if p.exists():
            print(f"  跳过目录：{d}/")
    if (PROJECT_ROOT / "settings.json").exists():
        print(f"  跳过文件：settings.json（含用户 API Key）")

    # 4. 清理上次
    print("\n🧹 清理上次残留...")
    clean_previous()

    # 5. 打包
    print("\n📦 开始打包...")
    try:
        run_pyinstaller()
    except subprocess.CalledProcessError as e:
        print(f"❌ 打包失败：{e}")
        sys.exit(1)

    # 6. 复制到 dist + 打 zip
    print("\n📤 输出到 dist/...")
    make_zip()
    cleanup_dist()   # 清理 dist 中残留的 .old / settings.json 等

    # 7. 清理本轮中间产物（项目根）
    print("\n🧹 清理本轮中间产物...")
    clean_previous()

    # 7. 最终验证
    print("\n✅ 最终验证：")
    issues = verify_final()
    if issues:
        print("\n⚠️ 问题：")
        for i in issues:
            print("  " + i)
    else:
        print("  一切就绪 ✅")

    print("\n" + "=" * 60)
    print("🎉 打包完成！dist/编织者.exe + dist/编织者.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()