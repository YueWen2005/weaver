import requests
import os
import sys
import time
import re
import json
from PIL import Image
from io import BytesIO

# Provider 适配器层：目前内置「硅基流动」，以后新增平台只需在 providers.py 加类
import providers

# 应用版本号（设置页 / 窗口标题展示，打包发布时同步更新）
VERSION = "1.0"

# 全局字体名：由 detect_font() 在窗口创建时探测（优先渲染清晰的「等线」，回退微软雅黑）
FONT_NAME = "Microsoft YaHei UI"

# ============= Windows 控制台编码兼容 =============
# 打包成 exe 后控制台可能是 GBK 编码，emoji（🃏✅❌🎉等）会导致 UnicodeEncodeError 崩溃。
# 强制 stdout/stderr 用 UTF-8 且 errors=replace：无法显示的字符替换为 ?，程序正常运行。
if sys.platform == 'win32':
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# ============= 配置 =============
# 隐私保护：不在源码/打包 exe 中硬编码真实 API Key
# Key 只保存在用户本机的 settings.json（首次启动会引导填写），打包时 settings.json 被 build.py 排除
API_KEY = ""   # 占位符——真正的 Key 由用户在 GUI 设置页填写并保存到 settings.json
URL = "https://api.siliconflow.cn/v1/images/generations"


def get_output_dir():
    """输出目录：基于 exe 自身目录（打包后便携），开发时使用脚本所在目录。
    这样无论从桌面/快捷方式/命令行启动，图片都存在 exe 同目录的 generated_cards/，不会跑错位置。"""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "generated_cards")
    os.makedirs(p, exist_ok=True)
    return p


OUTPUT_DIR = get_output_dir()   # 兼容旧代码（用函数版保持后续动态更新）


# ============= 生成函数 =============
def get_api_settings():
    """从 settings.json 读取 API 与生成参数（启动后用户可在设置中修改）。
    Key / URL / 模型 默认均为空——未配置时返回空字符串，调用方负责提示用户填写。"""
    cfg = load_settings()
    return {
        "provider": cfg.get("provider", "siliconflow"),
        "url": cfg.get("api_url", ""),
        "key": cfg.get("api_key", ""),
        "model": cfg.get("model", ""),
        "image_size": cfg.get("image_size", "1024x1024"),
        "num_inference_steps": int(cfg.get("num_inference_steps", 30)),
        "guidance_scale": float(cfg.get("guidance_scale", 5)),
        "rate_per_minute": float(cfg.get("rate_per_minute", 12)),
    }


def get_provider():
    """工厂：按 settings.json 的 provider 字段创建当前平台适配器实例"""
    return providers.create_provider(load_settings())


def get_request_delay():
    """根据 rate_per_minute 计算每次生成的间隔（秒）。60/rate，最小 1 秒"""
    s = get_api_settings()
    return max(1.0, 60.0 / max(1.0, s["rate_per_minute"]))


def generate_image(prompt, save_path, reference_image=None):
    """生成一张图片并保存到 save_path。返回 (成功?, 失败原因)；成功时原因为空字符串。
    reference_image: 可选参考图（路径/二进制），支持图生图的模型会用上。
    通过 Provider 适配器转发到当前平台（默认硅基流动）。"""
    provider = get_provider()

    # 隐私保护：未配置 API Key / URL 时不发起请求
    if not provider.key:
        return False, "未配置 API Key —— 请先在「设置 → AI 调用」中填写你的密钥"
    if not provider.url:
        return False, "未配置 API URL —— 请先在「设置 → AI 调用」中填写接口地址"

    return provider.generate(prompt, save_path, reference_image)


# ============= 提取提示词 =============
def extract_prompts_from_text(text_content, user_ignore_words, block_english=False):
    """从文件内容中提取提示词（以句号结尾的完整句子，忽略无中文内容，使用用户输入的屏蔽词，不去重）

    block_english=True 时：屏蔽英文（保证字体清晰）—— 删除句子中的英文字母/英文单词部分，
    只保留中文内容，避免 AI 生成乱码英文。
    """
    prompts = []

    # 按句号切分（支持中文句号。和英文句号.）
    sentences = re.split(r'[。.]', text_content)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 跳过不含中文的内容（如纯英文、数字、ID）
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', sentence))
        if not has_chinese:
            continue

        # 可选：屏蔽英文（保留中文，去掉英文单词/字母，保证字体清晰）
        if block_english:
            sentence = re.sub(r'[A-Za-z]+', '', sentence)

        # 去掉句子首尾的常见标点和空白，避免杂质
        sentence = sentence.strip('，,、；;：:""\'\'（）()【】[]《》「」… \t')

        # 合并句子内部的换行/多余空白为单个空格（跨行句子也能保持完整）
        sentence = re.sub(r'\s+', ' ', sentence)

        if not sentence:
            continue

        # 屏蔽词过滤（精确匹配整句）
        if sentence in user_ignore_words:
            continue

        prompts.append(sentence)

    return prompts


# ============= 工具函数 =============
def safe_filename(name, idx):
    """将提示词转为安全的文件名（去除 Windows 非法字符）"""
    safe = re.sub(r'[\\/:*?"<>|\s]+', '_', name).strip('._')[:60]
    return f"{idx:03d}_{safe}.png" if safe else f"{idx:03d}_card.png"


def get_unique_save_path(save_path):
    """防覆盖：目标文件已存在时自动追加序号，如 xxx.png → xxx_2.png → xxx_3.png"""
    if not os.path.exists(save_path):
        return save_path
    base, ext = os.path.splitext(save_path)
    n = 2
    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def parse_dnd_files(data):
    """解析 tkinterdnd2 拖入的文件路径字符串（可能含花括号包裹的空格路径）"""
    matches = re.findall(r'\{([^}]*)\}|([^\s{][^\s}]*)', data)
    return [a or b for a, b in matches]


def clean_prompt_text(name, ignore_words, block_eng=False):
    """清洗单条提示词（屏蔽词删除 + 可选屏蔽英文 + 清理残留标点）。
    返回清洗后的文本；若整条被删空返回空字符串。"""
    cleaned = name
    # 1) 屏蔽词：从提示词中删除（子串替换）
    for w in ignore_words:
        cleaned = cleaned.replace(w, '')
    # 2) 可选：屏蔽英文
    if block_eng:
        cleaned = re.sub(r'[A-Za-z]+', '', cleaned)
    # 3) 清理删除后残留的标点/空白（如 ",,"、", "、开头逗号等）
    # 注意：[] 必须放在字符类最前面（否则 ] 会提前闭合字符类）
    cleaned = re.sub(r'[]\[,，、；;：:（）()【】《》「」…\t]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ============= GUI =============
def get_tk_classes():
    """优先使用 tkinterdnd2 的 Tk（支持文件拖放），失败回退标准 tkinter"""
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        return TkinterDnD.Tk, DND_FILES, True
    except Exception:
        import tkinter as tk
        return tk.Tk, None, False


def clear_window(root):
    for w in root.winfo_children():
        w.destroy()


def fit_geometry(root, w, h):
    """窗口自适应：顶部贴屏幕最顶，底部贴任务栏上沿（垂直占满工作区，水平居中）"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    # 获取工作区（已排除任务栏）
    try:
        class RECT(wintypes.RECT):
            pass
        warea = RECT()
        user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(warea), 0)  # SPI_GETWORKAREA
        wa_left, wa_top = warea.left, warea.top
        wa_w = warea.right - warea.left
        wa_h = warea.bottom - warea.top
    except Exception:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        wa_left, wa_top, wa_w, wa_h = 0, 0, sw, sh - 40

    # 宽度：请求尺寸（DPI 感知后 tkinter 单位为物理像素，需把设计宽度按 DPI 缩放），
    # 限制在工作区内，水平居中
    try:
        import ctypes as _c
        scale = max(1.0, _c.windll.user32.GetDpiForWindow(root.winfo_id()) / 96.0)
    except Exception:
        scale = 1.0
    w = min(int(w * scale), wa_w - 20)
    x = wa_left + max(0, (wa_w - w) // 2)

    # 先映射窗口，实测外框高度（= 客户区 + 标题栏 + 边框，Tk 的 geometry 只算客户区）
    try:
        root.update_idletasks()
        root.update()  # 确保窗口已创建并映射，GetWindowRect 才准确
        hwnd = user32.GetAncestor(root.winfo_id(), 2)
        class RECT2(wintypes.RECT):
            pass
        wr = RECT2()
        user32.GetWindowRect(hwnd, ctypes.byref(wr))
        frame_h = wr.bottom - wr.top
        client_h = root.winfo_height()
        delta = max(0, frame_h - client_h)  # 标题栏 + 边框高度
    except Exception:
        delta = 31  # 经验值（96DPI 标准标题栏）

    # 垂直：外框高度填满工作区 → 客户区高度 = 工作区高 - 边框
    h = max(200, wa_h - delta - 1)
    # 顶部贴屏幕顶，底部自然落在任务栏上沿
    y = wa_top

    # 设定后再次实测，微调使外框底部精确贴任务栏上沿
    try:
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.update_idletasks()
        user32.GetWindowRect(hwnd, ctypes.byref(wr))
        real_bottom = wr.bottom
        if real_bottom != warea.bottom:
            diff = real_bottom - warea.bottom
            h = max(200, h - diff)
            root.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        root.geometry(f"{w}x{h}+{x}+{y}")


# ---------- 配置文件（窗口分辨率持久化） ----------
def get_config_path():
    """配置文件路径：exe 同目录（打包后）或项目目录（开发时）"""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "settings.json")


def load_settings():
    """读取配置文件，失败返回空字典"""
    try:
        with open(get_config_path(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(cfg):
    """保存配置文件（失败静默，不影响主流程）"""
    try:
        with open(get_config_path(), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


MAX_MODEL_HISTORY = 20   # 每个平台最多保留的模型历史条数


def record_model_usage(provider, model):
    """记录一次「成功使用」的模型到历史（settings.json 的 model_history，按平台分组）。
    去重（最近使用排前），每平台最多保留 MAX_MODEL_HISTORY 条；失败静默。"""
    model = (model or "").strip()
    provider = (provider or "").strip()
    if not model or not provider:
        return
    try:
        cfg = load_settings()
        history = cfg.get("model_history") or {}
        if not isinstance(history, dict):
            history = {}
        lst = history.get(provider) or []
        if not isinstance(lst, list):
            lst = []
        # 去重：移除旧条目，新条目插到最前
        lst = [m for m in lst if m != model]
        lst.insert(0, model)
        history[provider] = lst[:MAX_MODEL_HISTORY]
        cfg["model_history"] = history
        save_settings(cfg)
    except Exception:
        pass


def get_model_history(provider):
    """读取某平台的历史模型列表（最新在前）"""
    try:
        history = load_settings().get("model_history") or {}
        lst = history.get(provider) or []
        return [m for m in lst if isinstance(m, str) and m.strip()]
    except Exception:
        return []


def apply_window_size(root, default_w, default_h):
    """应用配置中的分辨率（未配置则用默认值），并做屏幕自适应"""
    w, h = default_w, default_h
    try:
        rsize = load_settings().get("window_size")
        if rsize:
            w, h = map(int, str(rsize).lower().split('x'))
    except Exception:
        pass
    fit_geometry(root, w, h)


def build_top_bar(root, title_text, right_btn=None):
    """构建顶部装饰条（金色条 + 标题文字 + 可选右侧按钮）"""
    import tkinter as tk

    FONT = FONT_NAME
    BG = "#0d0d1a"
    DIM = "#8a8aa8"
    GOLD = "#f0c75e"

    # 金色装饰条
    tk.Frame(root, bg=GOLD, height=4).pack(fill="x")

    # 标题行（装饰性文字，无按钮、无拖拽）
    bar = tk.Frame(root, bg=BG, height=30)
    bar.pack(fill="x", padx=14)
    bar.pack_propagate(False)
    tk.Label(bar, text=title_text, bg=BG, fg=DIM,
             font=(FONT, 9)).pack(side="left")

    # 可选右侧按钮（text, command）
    if right_btn:
        text, cmd = right_btn
        btn = tk.Button(bar, text=text, bg=BG, fg=DIM, relief="flat", bd=0,
                        font=(FONT, 9), cursor="hand2", command=cmd,
                        activebackground=BG, activeforeground=GOLD)
        btn.pack(side="right")
        btn.bind("<Enter>", lambda e: btn.configure(fg=GOLD))
        btn.bind("<Leave>", lambda e: btn.configure(fg=DIM))
    return bar


# ---------- 页面1：启动页 ----------
def create_app_window():
    """创建应用主窗口（系统原生窗口，任务栏/最小化/最大化原生支持）"""
    import tkinter as tk

    # ===== DPI 感知：根治高 DPI 屏幕下字体/界面模糊 =====
    # tkinter 默认不感知 DPI，Windows 会对整个界面做位图拉伸导致文字发虚。
    # 必须在创建 Tk 窗口之前启用进程级 DPI 感知。
    # 优先级：PerMonitorV2（最清晰）→ 系统级 v1 → 用户级，任一成功即可。
    # 注意：打包时 build.py 已通过 dpi_aware.manifest 在 exe 启动时声明 DPI 感知，
    # 这里是双保险（覆盖 manifest 未生效的极端情况）。
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE_V2 = -4（每显示器 v2，Win10 1703+）
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
    except Exception:
        try:
            import ctypes
            if hasattr(ctypes.windll, 'shcore'):
                # PROCESS_PER_MONITOR_DPI_AWARE = 2
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            else:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    TkClass, DND_FILES, has_dnd = get_tk_classes()

    BG = "#0d0d1a"

    root = TkClass()
    root.title(f"编织者 v{VERSION}")
    root.configure(bg=BG)
    # DPI 感知后按实际 DPI 调整 tk 缩放，保证字体物理大小与设计一致、渲染清晰
    try:
        import ctypes
        dpi = ctypes.windll.user32.GetDpiForWindow(root.winfo_id()) or 96
        root.tk.call('tk', 'scaling', dpi / 72.0)
    except Exception:
        pass
    # 字体探测：优先「等线 DengXian」（小字号渲染锐利），回退微软雅黑
    global FONT_NAME
    try:
        import tkinter.font as _tkfont
        families = set(_tkfont.families(root))
        if "DengXian" in families:
            FONT_NAME = "DengXian"
        elif "微软雅黑" in families or "Microsoft YaHei" in families:
            FONT_NAME = "Microsoft YaHei"
        else:
            FONT_NAME = "Microsoft YaHei UI"
    except Exception:
        pass
    # 尝试设置窗口图标（开发时用 .ico，打包后 exe 自带图标）
    try:
        if os.path.exists("编织者.ico"):
            root.iconbitmap("编织者.ico")
    except Exception:
        pass
    # 允许拉伸/最大化（resizable(False,False) 会禁用系统最大化按钮）
    # 设置最小尺寸避免被拖得太小
    root.minsize(560, 600)

    return root


def build_launch_page(root):
    """构建启动页内容（复用 root，返回时不再创建新窗口）"""
    import tkinter as tk
    from tkinter import messagebox

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    PURPLE = "#8b7cf6"
    DIM = "#8a8aa8"

    root.configure(bg=BG)
    apply_window_size(root, 580, 700)

    # ---- 顶部装饰条 + 标题行（右上角：历史记录入口） ----
    def open_history():
        clear_window(root)
        show_history_page(root)

    build_top_bar(root, "🃏 编织者 · 卡牌生成系统",
                  right_btn=("🕘 历史记录", open_history))

    # ---- 底部栏：版本号（左侧）+ 设置按钮（右侧）----
    # 注意：必须最先 pack(side="bottom")，否则内容溢出时设置按钮会被压缩到看不见
    bottom_bar = tk.Frame(root, bg=BG)
    bottom_bar.pack(side="bottom", fill="x", padx=18, pady=(0, 12))

    tk.Label(bottom_bar, text=f"v{VERSION}", bg=BG, fg="#7a7aa0",
             font=(FONT, 9)).pack(side="left")

    def open_settings():
        clear_window(root)
        show_settings_page(root)

    def settings_hover(on):
        settings_btn.configure(fg=GOLD if on else DIM)

    settings_btn = tk.Button(bottom_bar, text="⚙ 设置", bg="#23234a", fg=DIM,
                             font=(FONT, 10), relief="flat", bd=0,
                             cursor="hand2", padx=16, pady=6,
                             activebackground="#2e2e5e", activeforeground=GOLD,
                             command=open_settings)
    settings_btn.pack(side="right")
    settings_btn.bind("<Enter>", lambda e: settings_hover(True))
    settings_btn.bind("<Leave>", lambda e: settings_hover(False))

    # ---- 按钮区（固定在底部栏上方，先 pack(side="bottom") 保证按钮始终可见）----
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(side="bottom", pady=(0, 12))

    def on_start():
        clear_window(root)
        show_mode_page(root)

    def bind_hover(btn, hover_on=True):
        btn.configure(bg="#ffd98a" if hover_on else GOLD)

    start_btn = tk.Button(btn_frame, text="⚡ 开 始 生 成", bg=GOLD, fg="#14142b",
                          font=(FONT, 13, "bold"), relief="flat", bd=0,
                          padx=36, pady=10, cursor="hand2",
                          activebackground="#ffd98a", activeforeground="#14142b",
                          command=on_start)
    start_btn.pack(side="left", padx=8)
    start_btn.bind("<Enter>", lambda e: bind_hover(start_btn, True))
    start_btn.bind("<Leave>", lambda e: bind_hover(start_btn, False))

    quit_btn = tk.Button(btn_frame, text="退 出", bg="#23234a", fg=DIM,
                         font=(FONT, 11), relief="flat", bd=0,
                         padx=26, pady=10, cursor="hand2",
                         activebackground="#2e2e5e", activeforeground="#e8e8f0",
                         command=root.destroy)
    quit_btn.pack(side="left", padx=8)

    # ---- 卡牌 Logo（Canvas 绘制，缩小以适配小屏幕） ----
    logo = tk.Canvas(root, width=150, height=170, bg=BG, highlightthickness=0)
    logo.pack(pady=(10, 4))

    logo.create_rectangle(22, 10, 128, 160, fill="#1a1a3f", outline=GOLD, width=2)
    logo.create_rectangle(29, 17, 121, 153, outline="#3b3b6e", width=1)
    # 中央大星 + 光晕
    logo.create_text(75, 82, text="✦", fill=GOLD, font=(FONT, 36))
    for dx, dy in ((-18, -15), (20, -21), (15, 18), (-21, 20), (0, -35), (0, 36)):
        logo.create_oval(75 + dx - 3, 82 + dy - 3, 75 + dx + 3, 82 + dy + 3,
                         fill=PURPLE, outline="")
    # 上下小星
    logo.create_text(75, 34, text="✦", fill=GOLD, font=(FONT, 10))
    logo.create_text(75, 134, text="✦", fill=GOLD, font=(FONT, 10))
    # 卡面文字
    logo.create_text(75, 104, text="weaver", fill="#5a5a8f", font=(FONT, 7))

    # ---- 标题（缩小字号） ----
    tk.Label(root, text="编 织 者", bg=BG, fg=GOLD,
             font=(FONT, 24, "bold")).pack()
    tk.Label(root, text="卡 牌 图 片 批 量 生 成 器", bg=BG, fg="#e8e8f0",
             font=(FONT, 11)).pack(pady=(2, 8))

    # ---- 信息面板（每行 Frame，emoji 固定宽度 + 文字左对齐）----
    panel = tk.Frame(root, bg=PANEL, padx=24, pady=12)
    panel.pack(fill="x", padx=40, pady=(6, 14))

    # 显示当前设置（从 settings.json 实时读取，设置中修改后进入页面即同步）
    # 隐私保护：不显示完整 API URL / Key，仅显示配置状态
    _api_s = get_api_settings()
    _key_ok = bool(_api_s["key"].strip())
    _url_ok = bool(_api_s["url"].strip())
    if _key_ok and _url_ok:
        _api_status = "已配置 ✓"
        _api_color = "#e8e8f0"
    else:
        _api_status = "未配置 — 去设置填写"
        _api_color = "#ff6b6b"

    infos = [
        ("⚙️", "生成模型", _api_s["model"]),
        ("🖼️", "输出画幅", _api_s["image_size"]),
        ("📦", "生成方式", "批量 / 单个"),
        ("🔑", "API 账号", _api_status),
    ]
    for emoji, k, v in infos:
        row = tk.Frame(panel, bg=PANEL)
        row.pack(fill="x", pady=4)
        # emoji 单独 label，固定 3 字符宽度（吸收不同 emoji 的宽度差异）
        tk.Label(row, text=emoji, bg=PANEL, fg=DIM,
                 font=(FONT, 10), width=3, anchor="w").pack(side="left")
        # 中文词 label，起始位置固定，全部左对齐
        tk.Label(row, text=k, bg=PANEL, fg=DIM,
                 font=(FONT, 10), anchor="w").pack(side="left", padx=(2, 0))
        # 值右对齐到 panel 内右边界（API 状态行用彩色）
        tk.Label(row, text=v, bg=PANEL,
                 fg=_api_color if k == "API 账号" else "#e8e8f0",
                 font=(FONT, 10, "bold"), anchor="e").pack(side="right")

    # ---- 流程说明（可滚动：鼠标悬停后滚轮上下翻阅） ----
    steps = ("① 选择生成方式\n"
             "② 批量：拖入文本文件自动提取提示词\n"
             "③ 奇幻风格卡牌插画 → generated_cards/\n"
             "④ 生成完成后可到「历史记录」查看、复制或删除\n"
             "⑤ 在「设置」中可调整窗口分辨率适配屏幕")
    steps_text = tk.Text(root, height=3, wrap="word", bg=BG, fg=DIM,
                         font=(FONT, 10), relief="flat", bd=0,
                         highlightthickness=0, cursor="arrow")
    steps_text.insert("1.0", steps)
    steps_text.config(state="disabled")
    steps_text.pack(fill="x", padx=40, pady=(0, 8))

    def _steps_wheel(e):
        # 鼠标悬停在说明区时，滚轮上下翻阅
        steps_text.yview_scroll(-1 * (e.delta // 120), "units")
        return "break"

    steps_text.bind("<Enter>", lambda e: steps_text.focus_set())
    steps_text.bind("<MouseWheel>", _steps_wheel)


def _api_config_missing():
    """检查 API 配置是否缺失（首次启动时提示填写）"""
    cfg = load_settings()
    missing = []
    if not str(cfg.get("api_key", "")).strip():
        missing.append("API Key")
    if not str(cfg.get("api_url", "")).strip():
        missing.append("API URL")
    if not str(cfg.get("model", "")).strip():
        missing.append("模型名称")
    return missing


def require_api_setup(root):
    """首次启动：若无有效 API 配置，自动弹出配置窗口让用户填写自己的账号"""
    import tkinter as tk
    from tkinter import messagebox

    missing = _api_config_missing()
    if not missing:
        return

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    RED = "#ff6b6b"

    win = tk.Toplevel(root)
    win.title("首次配置 · 填写你的 API 账号")
    win.configure(bg=BG)
    win.transient(root)
    win.resizable(False, False)

    W, H = 540, 560
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    win.geometry(f"{W}x{H}+{max(0, (sw - W) // 2)}+{max(0, (sh - H) // 2)}")

    # 标题
    tk.Label(win, text="首次使用 · 请填写 API 账号", bg=BG, fg=GOLD,
             font=(FONT, 15, "bold")).pack(pady=(20, 4))
    tk.Label(win, text=f"缺少：{'、'.join(missing)}",
             bg=BG, fg=RED, font=(FONT, 10)).pack()

    # 表单
    panel = tk.Frame(win, bg=PANEL, padx=22, pady=14)
    panel.pack(fill="x", padx=36, pady=(12, 6))

    cfg = load_settings()
    vars_ = {}

    def row(label_text, key, show=None):
        tk.Label(panel, text=label_text, bg=PANEL, fg="#e8e8f0",
                 font=(FONT, 10), anchor="w").pack(fill="x", pady=(8, 2))
        # 默认始终为空，用户自己填写（隐私 + 不预填默认值）
        var = tk.StringVar(value=str(cfg.get(key, "") or ""))
        vars_[key] = var
        e = tk.Entry(panel, textvariable=var, bg="#1d1d42",
                     fg="#e8e8f0", insertbackground=GOLD,
                     font=(FONT, 10), relief="flat", bd=0,
                     highlightthickness=1, highlightbackground="#3b3b6e",
                     highlightcolor=GOLD, show=show)
        e.pack(fill="x", ipady=7)
        return e

    row("API URL（接口地址）", "api_url")
    row("API Key（密钥）", "api_key", show="•")
    row("模型名称（Model）", "model")

    tk.Label(win,
             text="参考：SiliconFlow 地址 https://api.siliconflow.cn/v1/images/generations\n"
                  "模型示例 Kwai-Kolors/Kolors · 在 SiliconFlow 控制台获取 Key\n"
                  "填写后点击「保存」，可在设置中随时修改。",
             bg=BG, fg="#7a7aa0", font=(FONT, 8), justify="center").pack(pady=(6, 4))

    # 按钮
    btns = tk.Frame(win, bg=BG)
    btns.pack(pady=(8, 16))

    def on_save():
        url = vars_["api_url"].get().strip()
        key = vars_["api_key"].get().strip()
        model = vars_["model"].get().strip()
        if not url or not key or not model:
            messagebox.showwarning("提示", "三项都必须填写完整")
            return
        cfg = load_settings()
        cfg["api_url"] = url
        cfg["api_key"] = key
        cfg["model"] = model
        save_settings(cfg)
        win.destroy()
        messagebox.showinfo("已保存", "API 账号已保存，可以开始使用了")

    def on_skip():
        # 跳过：不保存，用户稍后可在设置中填写
        win.destroy()

    tk.Button(btns, text="✅ 保 存", bg=GOLD, fg="#14142b",
              font=(FONT, 12, "bold"), relief="flat", bd=0,
              padx=30, pady=9, cursor="hand2",
              activebackground="#ffd98a", activeforeground="#14142b",
              command=on_save).pack(side="left", padx=8)
    tk.Button(btns, text="稍后再说", bg="#23234a", fg=DIM,
              font=(FONT, 11), relief="flat", bd=0,
              padx=22, pady=9, cursor="hand2",
              activebackground="#2e2e5e", activeforeground="#e8e8f0",
              command=on_skip).pack(side="left", padx=8)

    win.grab_set()   # 模态：必须先处理配置窗口


def show_launch_window():
    """创建主窗口并显示启动页"""
    root = create_app_window()
    build_launch_page(root)
    # 首次启动若无 API 配置，自动弹出配置窗口（等主窗口渲染完成后）
    root.after(300, lambda: require_api_setup(root))
    root.mainloop()



# ---------- 页面2：模式选择页 ----------
def show_mode_page(root):
    """选择生成方式：批量生成 / 单个生成"""
    import tkinter as tk

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"

    root.configure(bg=BG)
    apply_window_size(root, 580, 480)

    # ---- 顶部装饰条 + 标题行 ----
    build_top_bar(root, "🃏 编织者 · 选择生成方式")

    # ---- 标题 ----
    tk.Label(root, text="选 择 生 成 方 式", bg=BG, fg="#e8e8f0",
             font=(FONT, 16, "bold")).pack(pady=(28, 6))
    tk.Label(root, text="请选择你需要的生成模式", bg=BG, fg=DIM,
             font=(FONT, 10)).pack()

    # ---- 模式按钮 ----
    mode_frame = tk.Frame(root, bg=BG)
    mode_frame.pack(pady=(26, 10))
    # 让两列等宽：mode_frame 宽度由最宽卡片决定，强制卡片宽度一致
    mode_frame.grid_columnconfigure(0, uniform="a")
    mode_frame.grid_columnconfigure(1, uniform="a")

    def bind_hover(btn, on, bg_on):
        btn.configure(bg=bg_on if on else PANEL)

    def choose_batch():
        # 图生图模式（img）→ 直接在模式选择页拦截，根本不让用户进入批量页
        _api_s = get_api_settings()
        _cap = providers.resolve_image_input_capability(
            _api_s["provider"], _api_s["model"],
            overrides=load_settings().get("image_input_overrides", {}))
        if _cap == "img":
            messagebox.showinfo(
                "图生图模型暂不支持",
                f"当前模型「{_api_s['model']}」为图生图模式，不支持批量生成。\n"
                f"请使用「单个生成」模式（可上传参考图），\n"
                f"或在「设置 → AI 调用」中切换为「参考图兼容」模式后批量生成。")
            return
        clear_window(root)
        show_batch_page(root)

    def choose_single():
        clear_window(root)
        show_single_page(root)

    batch_btn = tk.Button(mode_frame, text="📦 批量生成", bg=PANEL, fg=GOLD,
                          font=(FONT, 14, "bold"), relief="flat", bd=0,
                          width=14, pady=18, cursor="hand2",
                          activebackground="#1d1d42", activeforeground=GOLD,
                          command=choose_batch)
    batch_btn.grid(row=0, column=0, padx=10)
    batch_btn.bind("<Enter>", lambda e: bind_hover(batch_btn, True, "#1d1d42"))
    batch_btn.bind("<Leave>", lambda e: bind_hover(batch_btn, False, PANEL))

    # 图生图模式：批量按钮视觉置灰 + hover 提示「不可用」
    try:
        _api_s_init = get_api_settings()
        _cap_init = providers.resolve_image_input_capability(
            _api_s_init["provider"], _api_s_init["model"],
            overrides=load_settings().get("image_input_overrides", {}))
        if _cap_init == "img":
            batch_btn.config(fg="#4a4a6a", cursor="X_cursor",
                             text="📦 批量生成 (图生图不可用)")
            batch_btn.unbind("<Enter>")
            batch_btn.unbind("<Leave>")
    except Exception:
        pass

    single_btn = tk.Button(mode_frame, text="🖼️ 单个生成", bg=PANEL, fg=DIM,
                           font=(FONT, 14, "bold"), relief="flat", bd=0,
                           width=14, pady=18, cursor="hand2",
                           activebackground="#1d1d42", activeforeground=DIM,
                           command=choose_single)
    single_btn.grid(row=0, column=1, padx=10)
    single_btn.bind("<Enter>", lambda e: bind_hover(single_btn, True, "#1d1d42"))
    single_btn.bind("<Leave>", lambda e: bind_hover(single_btn, False, PANEL))

    # 模式说明卡片（清晰区分两种模式的适用场景）
    def make_desc(parent, title, points, row, col, accent):
        # 固定宽度 220px（配合 grid uniform="a" 让两卡片等宽）
        card = tk.Frame(parent, bg=PANEL, padx=16, pady=12, width=220)
        card.grid(row=row, column=col, padx=10, pady=(12, 0), sticky="nsew")
        card.grid_propagate(False)
        # 标题（金色加粗，与下方要点明确分隔）
        tk.Label(card, text=title, bg=PANEL, fg=accent,
                 font=(FONT, 10, "bold"), anchor="w").pack(fill="x", pady=(0, 6))
        # 分隔线（金色细线，视觉强化卡片结构）
        sep = tk.Frame(card, bg=accent, height=1)
        sep.pack(fill="x", pady=(0, 8))
        for p in points:
            tk.Label(card, text=p, bg=PANEL, fg=DIM,
                     font=(FONT, 9), anchor="w", justify="left",
                     wraplength=200).pack(fill="x", pady=(3, 0))

    make_desc(mode_frame, "适合：一次生成多张",
              ["• 拖入一个 .txt 文本文件",
               "• 自动按句号提取提示词",
               "• 逐张批量生成，适合作大量制作"],
              row=1, col=0, accent=GOLD)
    make_desc(mode_frame, "适合：单独制作一张",
              ["• 手动输入一个提示词",
               "• 只生成这一张图",
               "• 试效果或单独定制时用"],
              row=1, col=1, accent=DIM)

    # ---- 返回按钮 ----
    def go_back():
        clear_window(root)
        build_launch_page(root)   # 复用同一窗口，不创建新 root

    back_btn = tk.Button(root, text="← 返回", bg="#23234a", fg=DIM,
                         font=(FONT, 10), relief="flat", bd=0,
                         padx=16, pady=6, cursor="hand2",
                         activebackground="#2e2e5e", activeforeground="#e8e8f0",
                         command=go_back)
    back_btn.pack(side="bottom", pady=(0, 16))


# ---------- 页面3：批量生成页（拖入文本文件） ----------
def ask_ignore_words(root):
    """开始生成前弹窗：让用户输入屏蔽词（逗号分隔，可留空跳过）。
    返回屏蔽词列表；用户点「取消」返回 None。"""
    import tkinter as tk
    from tkinter import messagebox

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"

    result = {"value": None}   # 保存结果，避免 lambda 捕获循环

    win = tk.Toplevel(root)
    win.title("屏蔽词设置")
    win.configure(bg=BG)
    win.transient(root)
    win.resizable(False, False)

    W, H = 480, 280
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    win.geometry(f"{W}x{H}+{max(0, (sw - W) // 2)}+{max(0, (sh - H) // 2)}")

    tk.Label(win, text="📌 设置屏蔽词（可选）", bg=BG, fg=GOLD,
             font=(FONT, 13, "bold")).pack(pady=(18, 4))
    tk.Label(win, text="这些词会从提示词中删除后再生成，多个词用逗号分隔",
             bg=BG, fg=DIM, font=(FONT, 10)).pack()
    tk.Label(win, text="例如：生成一只猫 → 提示词中的\"生成一只猫\"会被去掉",
             bg=BG, fg="#7a7aa0", font=(FONT, 9)).pack(pady=(2, 0))

    entry = tk.Entry(win, bg="#1d1d42", fg="#e8e8f0", insertbackground=GOLD,
                     font=(FONT, 11), relief="flat", bd=0,
                     highlightthickness=1, highlightbackground="#3b3b6e",
                     highlightcolor=GOLD)
    entry.pack(fill="x", padx=50, ipady=6, pady=(12, 6))

    btns = tk.Frame(win, bg=BG)
    btns.pack(pady=(6, 14))

    def on_ok():
        raw = entry.get().strip()
        words = [w.strip() for w in raw.split(',') if w.strip()]
        result["value"] = words
        win.destroy()

    def on_cancel():
        result["value"] = None
        win.destroy()

    tk.Button(btns, text="✅ 开始生成", bg=GOLD, fg="#14142b",
              font=(FONT, 11, "bold"), relief="flat", bd=0,
              padx=22, pady=7, cursor="hand2",
              activebackground="#ffd98a", activeforeground="#14142b",
              command=on_ok).pack(side="left", padx=8)
    tk.Button(btns, text="取消", bg="#23234a", fg=DIM,
              font=(FONT, 10), relief="flat", bd=0,
              padx=18, pady=7, cursor="hand2",
              activebackground="#2e2e5e", activeforeground="#e8e8f0",
              command=on_cancel).pack(side="left", padx=8)

    entry.bind("<Return>", lambda e: on_ok())
    win.grab_set()
    win.wait_window()   # 阻塞直到窗口关闭

    return result["value"]


def ask_test_mode(root):
    """点击「测试连接」时的方式选择弹窗。
    返回 'full'（真实请求测试）/ 'format'（仅格式检查）/ None（取消）。"""
    import tkinter as tk

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    RED = "#ff6b6b"
    GREEN = "#5cd68b"

    result = {"mode": None}

    win = tk.Toplevel(root)
    win.title("测试连接 · 选择方式")
    win.configure(bg=BG)
    win.transient(root)
    win.resizable(False, False)

    W, H = 540, 440
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    win.geometry(f"{W}x{H}+{max(0, (sw - W) // 2)}+{max(0, (sh - H) // 2)}")

    tk.Label(win, text="🔌 测试连接", bg=BG, fg=GOLD,
             font=(FONT, 13, "bold")).pack(pady=(14, 4))
    tk.Label(win, text="点击下方选择测试方式：", bg=BG, fg="#e8e8f0",
             font=(FONT, 10)).pack()

    def pick(mode):
        result["mode"] = mode
        win.destroy()

    def make_clickable_card(parent, title, desc, accent_color, on_click):
        """把整张卡片做成可点击区域：hover 高亮、点击触发 on_click。"""
        card = tk.Frame(parent, bg=PANEL, padx=14, pady=10, cursor="hand2")
        widgets = []
        def on_enter(_):
            card.config(bg="#1d1d42")
            for w in widgets:
                w.config(bg="#1d1d42")
        def on_leave(_):
            card.config(bg=PANEL)
            for w in widgets:
                w.config(bg=PANEL)
        # 用 lambda 绑定避免内部 def 命名与外层 on_click 参数冲突
        card.bind("<Button-1>", lambda _: on_click())
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

        lbl_title = tk.Label(card, text=title, bg=PANEL, fg=accent_color,
                            font=(FONT, 11, "bold"), anchor="w", cursor="hand2")
        lbl_title.pack(fill="x")
        lbl_desc = tk.Label(card, text=desc, bg=PANEL, fg=DIM,
                            font=(FONT, 9), justify="left", anchor="w",
                            cursor="hand2", wraplength=440)
        lbl_desc.pack(fill="x", pady=(4, 0))
        widgets.extend([lbl_title, lbl_desc])
        for w in widgets:
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", lambda _: on_click())
        return card

    # 两种方式：卡片本身就是按钮（点击即选，无需底部按钮区）
    make_clickable_card(
        win,
        "① 完整测试（推荐）",
        "发送一次最小请求（1 步）到平台，真实验证 Key / 模型 / 网络是否可用。\n"
        "会消耗约 1/30 张的生成额度。",
        GOLD, lambda: pick("full"),
    ).pack(fill="x", padx=36, pady=(10, 6))
    make_clickable_card(
        win,
        "② 仅格式检查（免费）",
        "不发请求、不花额度，只检查 URL / Key / 模型名格式是否填写正确。\n"
        "适合只想确认填写有没有漏。",
        GREEN, lambda: pick("format"),
    ).pack(fill="x", padx=36, pady=(0, 6))

    # 取消提示（点窗口右上角 ✕ 或按 ESC）
    tk.Label(win, text="点击上方选择方式 · 按 ESC 或关闭窗口取消",
             bg=BG, fg="#7a7aa0", font=(FONT, 9)).pack(pady=(6, 12))
    win.bind("<Escape>", lambda e: pick(None))

    win.grab_set()
    win.wait_window()
    return result["mode"]


def ask_reference_image(root, required=False):
    """生成开始前：当模型支持参考图时，让用户选择参考图。
    required=False：可选，可跳过（返回 None = 跳过/取消）。
    required=True：必须选择（图生图模型），无「跳过」按钮；取消返回 None（调用方终止生成）。"""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    GREEN = "#5cd68b"

    result = {"path": None}

    win = tk.Toplevel(root)
    win.title("选择参考图")
    win.configure(bg=BG)
    win.transient(root)
    win.resizable(False, False)

    W, H = 500, 270
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    win.geometry(f"{W}x{H}+{max(0, (sw - W) // 2)}+{max(0, (sh - H) // 2)}")

    title_text = "🖼️ 选择参考图（必选）" if required else "🖼️ 选择参考图（可选）"
    tk.Label(win, text=title_text, bg=BG, fg=GOLD,
             font=(FONT, 13, "bold")).pack(pady=(18, 4))
    desc_text = ("当前模型为图生图模型，必须提供一张参考图" if required
                 else "当前模型支持图生图，可提供一张参考图作为风格/内容依据")
    tk.Label(win, text=desc_text, bg=BG, fg=DIM,
             font=(FONT, 10)).pack()

    path_label = tk.Label(win,
                          text="尚未选择参考图" if required else "未选择参考图（将按纯文生图生成）",
                          bg=BG, fg="#7a7aa0", font=(FONT, 9), wraplength=420)
    path_label.pack(pady=(12, 2))

    def choose_img():
        p = filedialog.askopenfilename(
            title="选择参考图",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp"), ("所有文件", "*.*")])
        if p:
            result["path"] = p
            path_label.config(text=f"✅ 参考图：{os.path.basename(p)}", fg=GREEN)

    tk.Button(win, text="📁 选择参考图", bg="#23234a", fg="#e8e8f0",
              font=(FONT, 11), relief="flat", bd=0,
              padx=20, pady=8, cursor="hand2",
              activebackground="#2e2e5e",
              command=choose_img).pack(pady=(6, 4))

    btns = tk.Frame(win, bg=BG)
    btns.pack(pady=(10, 14))

    def on_ok():
        if required and not result["path"]:
            messagebox.showwarning("提示", "请先选择参考图")
            return
        win.destroy()

    def on_skip():
        result["path"] = None   # 跳过/取消
        win.destroy()

    tk.Button(btns, text="✅ 确 定", bg=GOLD, fg="#14142b",
              font=(FONT, 11, "bold"), relief="flat", bd=0,
              padx=22, pady=7, cursor="hand2",
              activebackground="#ffd98a", activeforeground="#14142b",
              command=on_ok).pack(side="left", padx=8)
    if not required:
        tk.Button(btns, text="跳过（纯文生图）", bg="#23234a", fg=DIM,
                  font=(FONT, 10), relief="flat", bd=0,
                  padx=18, pady=7, cursor="hand2",
                  activebackground="#2e2e5e", activeforeground="#e8e8f0",
                  command=on_skip).pack(side="left", padx=8)
    else:
        tk.Button(btns, text="取消生成", bg="#23234a", fg=DIM,
                  font=(FONT, 10), relief="flat", bd=0,
                  padx=18, pady=7, cursor="hand2",
                  activebackground="#2e2e5e", activeforeground="#e8e8f0",
                  command=on_skip).pack(side="left", padx=8)

    win.grab_set()
    win.wait_window()
    return result["path"]


def show_batch_page(root):
    """拖入 .txt 文本文件，自动提取提示词并批量生成"""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    TkClass, DND_FILES, has_dnd = get_tk_classes()

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    GREEN = "#5cd68b"
    RED = "#ff6b6b"

    root.configure(bg=BG)
    apply_window_size(root, 580, 560)

    state = {"file_path": None, "prompts": [],
             "gen_thread": None, "cancel": False,
             "progress": "", "done": False, "success": 0, "total": 0}

    # ---- 顶部装饰条 + 标题行 ----
    build_top_bar(root, "📦 批量生成")

    # ---- 标题 ----
    tk.Label(root, text="批 量 生 成", bg=BG, fg=GOLD,
             font=(FONT, 18, "bold")).pack(pady=(20, 4))
    tk.Label(root, text="拖入 .txt 文本文件，自动提取句号结尾的提示词",
             bg=BG, fg=DIM, font=(FONT, 10)).pack()

    # ---- 拖放区 ----
    drop_frame = tk.Frame(root, bg=PANEL, padx=20, pady=26)
    drop_frame.pack(fill="x", padx=50, pady=(18, 12))

    drop_label = tk.Label(drop_frame, text="📂 将文本文件拖入此处\n\n或点击下方按钮选择文件",
                          bg=PANEL, fg=DIM, font=(FONT, 12),
                          justify="center", cursor="hand2")
    drop_label.pack(fill="x")

    # ---- 文件信息 ----
    file_label = tk.Label(root, text="未选择文件", bg=BG, fg=DIM,
                          font=(FONT, 10))
    file_label.pack(pady=(4, 2))

    count_label = tk.Label(root, text="", bg=BG, fg=GREEN, font=(FONT, 10))
    count_label.pack()

    # ---- 选择文件按钮 ----
    def choose_file():
        path = filedialog.askopenfilename(
            title="选择文本文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
        if path:
            on_file_selected(path)

    def on_file_selected(path):
        """读取文件 → 提取提示词 → 更新界面"""
        # 1) 文件大小检查：空文件直接提示
        try:
            fsize = os.path.getsize(path)
        except OSError:
            fsize = 0
        if fsize == 0:
            state["file_path"] = None
            state["prompts"] = []
            file_label.config(text=f"📄 {os.path.basename(path)}", fg="#e8e8f0")
            count_label.config(text="❌ 文件是空的，请先在里面写入内容", fg="#ff6b6b")
            start_btn.config(state="disabled")
            messagebox.showerror("提示词为空", "文件是空的，请先在文件中写入提示词内容")
            return

        # 2) 读取内容：优先 UTF-8，失败则尝试 GBK（Windows 记事本默认编码）
        content = None
        for enc in ('utf-8', 'gbk'):
            try:
                with open(path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if content is None:
            state["file_path"] = None
            state["prompts"] = []
            file_label.config(text=f"📄 {os.path.basename(path)}", fg="#e8e8f0")
            count_label.config(text="❌ 无法识别文件编码（请用记事本另存为 UTF-8）", fg="#ff6b6b")
            start_btn.config(state="disabled")
            messagebox.showerror("读取失败", "无法识别文件编码，请用记事本另存为 UTF-8 后再试")
            return

        # 3) 提取提示词
        prompts = extract_prompts_from_text(content, [])

        state["file_path"] = path
        state["prompts"] = prompts

        file_label.config(text=f"📄 {os.path.basename(path)}", fg="#e8e8f0")
        if prompts:
            count_label.config(text=f"✅ 提取到 {len(prompts)} 个提示词", fg=GREEN)
            start_btn.config(state="normal")
        else:
            count_label.config(
                text="❌ 未提取到提示词：文件中需要「中文句子 + 句号」",
                fg="#ff6b6b")
            start_btn.config(state="disabled")
            messagebox.showerror(
                "提示词为空",
                "未从文件中提取到任何提示词。\n"
                "文件中需要「中文句子 + 句号（。或.）」格式，如：\n"
                "生成一个帅哥。生成一只猫。")

    def on_drop(event):
        if not has_dnd:
            return
        files = parse_dnd_files(event.data)
        txts = [f for f in files if f.lower().endswith('.txt')]
        if not txts:
            messagebox.showwarning("文件类型", "请拖入 .txt 文本文件")
            return
        on_file_selected(txts[0])

    # DND 注册
    if has_dnd:
        for w in (drop_frame, drop_label):
            w.drop_target_register(DND_FILES)
            w.dnd_bind('<<Drop>>', on_drop)

    # 点击拖放区也可选择文件
    for w in (drop_frame, drop_label):
        w.bind("<Button-1>", lambda e: choose_file())

    select_btn = tk.Button(root, text="📁 选择文件", bg="#23234a", fg="#e8e8f0",
                           font=(FONT, 11), relief="flat", bd=0,
                           padx=20, pady=8, cursor="hand2",
                           activebackground="#2e2e5e",
                           command=choose_file)
    select_btn.pack(pady=(6, 4))

    # ---- 选项区：屏蔽英文 + 画风 ----
    opt_panel = tk.Frame(root, bg=PANEL, padx=18, pady=10)
    opt_panel.pack(fill="x", padx=50, pady=(8, 4))

    # 屏蔽英文勾选
    block_eng_var = tk.BooleanVar(value=False)

    def toggle_english():
        """勾选/取消「屏蔽英文」后重新提取提示词"""
        if state["file_path"]:
            # 用当前勾选状态重新提取，更新计数
            content = None
            for enc in ('utf-8', 'gbk'):
                try:
                    with open(state["file_path"], 'r', encoding=enc) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if content is not None:
                prompts = extract_prompts_from_text(
                    content, [], block_english=block_eng_var.get())
                state["prompts"] = prompts
                count_label.config(
                    text=f"✅ 提取到 {len(prompts)} 个提示词" + ("（已屏蔽英文）" if block_eng_var.get() else ""),
                    fg=GREEN)
                start_btn.config(state="normal" if prompts else "disabled")

    chk = tk.Checkbutton(opt_panel, text="屏蔽英文",
                         variable=block_eng_var, bg=PANEL, fg="#e8e8f0",
                         selectcolor="#1d1d42", font=(FONT, 10),
                         activebackground=PANEL, activeforeground=GOLD,
                         highlightthickness=0, bd=0, anchor="w",
                         cursor="hand2", command=toggle_english)
    chk.pack(anchor="w", pady=(0, 6))

    # 画风输入
    style_label = tk.Label(opt_panel, text="画风（可选，默认：奇幻卡牌插画）",
                           bg=PANEL, fg="#e8e8f0", font=(FONT, 10), anchor="w")
    style_label.pack(fill="x")
    style_var = tk.StringVar(value="")
    style_entry = tk.Entry(opt_panel, textvariable=style_var, bg="#1d1d42",
                           fg="#e8e8f0", insertbackground=GOLD,
                           font=(FONT, 10), relief="flat", bd=0,
                           highlightthickness=1, highlightbackground="#3b3b6e",
                           highlightcolor=GOLD)
    style_entry.pack(fill="x", ipady=6)
    tk.Label(opt_panel, text="💡 例如：水墨画风 / 赛博朋克 / 宫崎骏动画风",
             bg=PANEL, fg=GOLD, font=(FONT, 8)).pack(anchor="w", pady=(4, 0))

    # ---- 进度状态 ----
    status_label = tk.Label(root, text="", bg=BG, fg=DIM, font=(FONT, 10))
    status_label.pack(pady=(10, 2))

    # ---- 操作按钮 ----
    op_frame = tk.Frame(root, bg=BG)
    op_frame.pack(pady=(10, 14))

    def go_back():
        # 若正在生成，先请求取消，等线程退出后再切页（避免卡顿/线程触碰已销毁组件）
        if state["gen_thread"] and state["gen_thread"].is_alive():
            state["cancel"] = True
            status_label.config(text="⏹ 正在停止…请稍候", fg=RED)
            root.after(150, _wait_back)
            return
        clear_window(root)
        show_mode_page(root)

    def _wait_back():
        if state["gen_thread"] and state["gen_thread"].is_alive():
            root.after(150, _wait_back)
            return
        clear_window(root)
        show_mode_page(root)

    def _poll_progress():
        """轮询后台线程进度，更新 UI"""
        st = state
        if st["gen_thread"] and st["gen_thread"].is_alive():
            if st["progress"]:
                status_label.config(text=st["progress"], fg="#e8e8f0")
            root.after(200, _poll_progress)
        else:
            # 线程已结束
            if st["done"]:
                failures = st.get("failures", [])
                status_label.config(
                    text=f"🎉 完成！成功生成 {st['success']}/{st['total']} 张"
                         + (f" · 失败 {len(failures)} 张" if failures else ""),
                    fg=GREEN)
                start_btn.config(state="normal")
                select_btn.config(state="normal")
                st["gen_thread"] = None
                # 如果有失败，弹出可滚动报告窗口
                if failures:
                    show_batch_report(root, st["success"], st["total"], failures)
            elif st["cancel"]:
                status_label.config(text="⏹ 已取消", fg=RED)
                start_btn.config(state="normal")
                select_btn.config(state="normal")
                st["gen_thread"] = None

    def on_start():
        """批量生成（后台线程执行，UI 轮询进度，返回按钮随时可点）"""
        import threading

        path = state.get("file_path")
        prompts = state.get("prompts", [])
        if not path or not prompts:
            messagebox.showerror("提示词为空", "请先选择文本文件并确认已提取到提示词")
            return

        # ---- 图生图模型拦截（在屏蔽词弹窗前，避免多余弹窗）----
        # 仅「图生图模式」（img）拦截批量；「参考图兼容」（both）允许批量（走文生图）
        _api_s = get_api_settings()
        _pid = _api_s["provider"]
        _model = _api_s["model"]
        _cap = providers.resolve_image_input_capability(
            _pid, _model, overrides=load_settings().get("image_input_overrides", {}))
        if _cap == "img":
            messagebox.showinfo(
                "图生图模型暂不支持",
                f"当前模型「{_model}」为图生图模式，暂不支持批量生成。\n"
                f"请使用「单个生成」模式（可上传参考图），\n"
                f"或切换为「参考图兼容」模式后批量生成（将按文生图处理）。")
            return

        # 开始前弹窗：提示输入屏蔽词（可跳过）
        ignore_words = ask_ignore_words(root)
        if ignore_words is None:      # 用户点了取消
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        state["cancel"] = False
        state["done"] = False
        state["success"] = 0
        state["total"] = len(prompts)
        state["progress"] = "准备中…"
        state["failures"] = []   # 收集失败 (提示词, 原因)

        start_btn.config(state="disabled")
        select_btn.config(state="disabled")

        # 收集画风（用户可输入自定义，默认奇幻卡牌插画）
        user_style = style_var.get().strip()
        style_suffix = user_style if user_style else "奇幻卡牌插画，精美"
        block_eng = block_eng_var.get()

        # ---- 预计算过滤后的提示词 ----
        # 屏蔽词 = 从提示词中「删除」的子串（如"生成一个帅哥,生成一只猫"+屏蔽"生成一只猫"→"生成一个帅哥"），
        # 删除后为空（整条都被删掉）则跳过该条；再叠加屏蔽英文清洗。
        valid_prompts = []
        for name in prompts:
            cleaned = clean_prompt_text(name, ignore_words, block_eng)
            if cleaned:
                valid_prompts.append(cleaned)

        # 过滤后为空：明确提示，不启动生成线程
        if not valid_prompts:
            status_label.config(text="⚠ 没有可生成的提示词（可能全部被屏蔽词过滤）", fg=RED)
            messagebox.showwarning(
                "提示",
                f"没有可生成的提示词。\n"
                f"原提示词 {len(prompts)} 条，全部被屏蔽词删除（{', '.join(ignore_words) or '空'}）或清洗后为空。\n"
                f"请检查屏蔽词设置或文本内容。")
            start_btn.config(state="normal")
            select_btn.config(state="normal")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        state["cancel"] = False
        state["done"] = False
        state["success"] = 0
        state["total"] = len(valid_prompts)
        state["progress"] = "准备中…"
        state["failures"] = []   # 收集失败 (提示词, 原因)

        start_btn.config(state="disabled")
        select_btn.config(state="disabled")

        def worker():
            success = 0
            failures = []
            total = len(valid_prompts)
            for idx, name in enumerate(valid_prompts, 1):
                if state["cancel"]:
                    break
                prompt = f"{style_suffix}，{name}"
                save_path = get_unique_save_path(
                    os.path.join(OUTPUT_DIR, safe_filename(name, idx)))
                state["progress"] = f"⏳ 正在生成 [{idx}/{total}] {name}"
                ok, err = generate_image(prompt, save_path)   # 批量纯文生图（图生图已在上方拦截）
                if ok:
                    success += 1
                    # 记录成功使用的模型到历史（供设置页下拉选择）
                    _a = get_api_settings()
                    record_model_usage(_a["provider"], _a["model"])
                else:
                    failures.append((name, err))
                state["success"] = success
                # 按 rate_per_minute 设置间隔
                if not state["cancel"] and idx < total:
                    time.sleep(get_request_delay())
            state["failures"] = failures
            state["done"] = True

        state["gen_thread"] = threading.Thread(target=worker, daemon=True)
        state["gen_thread"].start()
        root.after(200, _poll_progress)

    start_btn = tk.Button(op_frame, text="⚡ 开 始 生 成", bg=GOLD, fg="#14142b",
                          font=(FONT, 12, "bold"), relief="flat", bd=0,
                          padx=30, pady=10, cursor="hand2",
                          activebackground="#ffd98a", activeforeground="#14142b",
                          state="disabled", command=on_start)
    start_btn.pack(side="left", padx=8)

    back_btn = tk.Button(op_frame, text="← 返回", bg="#23234a", fg=DIM,
                         font=(FONT, 11), relief="flat", bd=0,
                         padx=22, pady=10, cursor="hand2",
                         activebackground="#2e2e5e", activeforeground="#e8e8f0",
                         command=go_back)
    back_btn.pack(side="left", padx=8)


def show_batch_report(root, success, total, failures):
    """批量生成完成报告：成功/失败统计 + 可滚动的失败明细列表"""
    import tkinter as tk

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    RED = "#ff6b6b"
    GREEN = "#5cd68b"

    win = tk.Toplevel(root)
    win.title(f"生成报告 · 成功 {success}/{total}")
    win.configure(bg=BG)
    win.transient(root)
    win.geometry("520x420")
    win.minsize(420, 300)

    # ---- 标题行 ----
    header = tk.Frame(win, bg=BG)
    header.pack(fill="x", padx=20, pady=(16, 8))
    tk.Label(header, text=f"🎉 成功 {success} / {total}",
             bg=BG, fg=GREEN, font=(FONT, 14, "bold")).pack(side="left")
    tk.Label(header, text=f"❌ 失败 {len(failures)}",
             bg=BG, fg=RED, font=(FONT, 14, "bold")).pack(side="right")

    # ---- 可滚动失败明细 ----
    body = tk.Frame(win, bg=PANEL, padx=2, pady=2)
    body.pack(fill="both", expand=True, padx=20, pady=(0, 12))

    text = tk.Text(body, wrap="word", bg=PANEL, fg="#e8e8f0",
                   font=(FONT, 10), relief="flat", bd=0,
                   highlightthickness=0, cursor="arrow")
    scroll = tk.Scrollbar(body, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)

    text.tag_configure("name", foreground=GOLD, font=(FONT, 10, "bold"))
    text.tag_configure("err",  foreground=RED)
    text.tag_configure("dim",  foreground=DIM)

    text.insert("end", f"失败明细（{len(failures)} 项，鼠标悬停后滚轮可上下滚动）：\n\n", "dim")
    for i, (name, err) in enumerate(failures, 1):
        text.insert("end", f"{i:>3}. ", "dim")
        text.insert("end", f"{name}\n", "name")
        text.insert("end", f"     ↳ {err}\n\n", "err")
    text.config(state="disabled")

    def _wheel(e):
        text.yview_scroll(-1 * (e.delta // 120), "units")
        return "break"

    text.bind("<Enter>", lambda e: text.focus_set())
    text.bind("<MouseWheel>", _wheel)

    # ---- 关闭按钮 ----
    tk.Button(win, text="关闭", bg=GOLD, fg="#14142b",
              font=(FONT, 11, "bold"), relief="flat", bd=0,
              padx=28, pady=8, cursor="hand2",
              activebackground="#ffd98a", activeforeground="#14142b",
              command=win.destroy).pack(pady=(0, 16))
    win.grab_set()


# ---------- 页面4：历史记录页（图片管理） ----------
PAGE_SIZE = 8  # 每页显示 8 张


def list_images():
    """列出 generated_cards 目录下的图片（按修改时间倒序，最新在前）"""
    imgs = []
    if not os.path.isdir(OUTPUT_DIR):
        return imgs
    exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
    for fn in os.listdir(OUTPUT_DIR):
        if fn.lower().endswith(exts):
            imgs.append(os.path.join(OUTPUT_DIR, fn))
    imgs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return imgs


def copy_image_to_clipboard(image_path):
    """复制图片到剪贴板（CF_DIB 位图格式），可直接粘贴到画图/微信/PowerPoint 等任何应用为图片内容"""
    import ctypes
    import io
    from ctypes import wintypes
    from PIL import Image as _Img

    if not image_path or not os.path.exists(image_path):
        return False
    try:
        img = _Img.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 保存为 BMP 取位图数据（CF_DIB 不含 BMP 文件头 14 字节）
        buf = io.BytesIO()
        img.save(buf, 'bmp')
        dib_data = buf.getvalue()[14:]

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        CF_DIB = 8
        GMEM_MOVEABLE = 0x0002

        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE

        hMem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib_data))
        if not hMem:
            return False
        pMem = kernel32.GlobalLock(hMem)
        if not pMem:
            kernel32.GlobalFree(hMem)
            return False
        ctypes.memmove(pMem, dib_data, len(dib_data))
        kernel32.GlobalUnlock(hMem)

        if not user32.OpenClipboard(None):
            kernel32.GlobalFree(hMem)
            return False
        user32.EmptyClipboard()
        user32.SetClipboardData(CF_DIB, hMem)
        user32.CloseClipboard()
        return True
    except Exception:
        return False


def show_history_page(root):
    """历史记录页：分页网格（8张/页）+ 批量删除 + 打开文件夹 + 单图检视"""
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    GREEN = "#5cd68b"
    RED = "#ff6b6b"
    THUMB = 128  # 缩略图边长

    state = {"page": 0}
    thumb_refs = []   # 保持 PhotoImage 引用防 GC
    page_paths = []   # 当前页图片路径（与勾选一一对应）
    check_vars = []   # 当前页勾选状态（与 page_paths 一一对应）

    root.configure(bg=BG)
    apply_window_size(root, 620, 700)

    def go_back():
        clear_window(root)
        build_launch_page(root)

    build_top_bar(root, "🕘 历史记录", right_btn=("← 返回", go_back))

    # ---- 工具行 ----
    toolbar = tk.Frame(root, bg=BG)
    toolbar.pack(fill="x", padx=20, pady=(10, 4))

    info_var = tk.StringVar(value="")
    tk.Label(toolbar, textvariable=info_var, bg=BG, fg=DIM,
             font=(FONT, 9)).pack(side="left")

    def open_folder():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            os.startfile(OUTPUT_DIR)
        except Exception as e:
            messagebox.showerror("打开失败", f"无法打开文件夹：\n{e}")

    def batch_delete():
        sel = [p for p, var in zip(page_paths, check_vars) if var.get()]
        if not sel:
            messagebox.showinfo("提示", "请先勾选要删除的图片")
            return
        delete_files(sel)

    # 批量删除按钮 + 打开文件夹按钮（固定创建一次）
    btn_frame = tk.Frame(toolbar, bg=BG)
    btn_frame.pack(side="right")
    tk.Button(btn_frame, text="🗑 批量删除", bg="#3a1f2e", fg=RED,
              font=(FONT, 10), relief="flat", bd=0, padx=14, pady=6,
              cursor="hand2", activebackground="#4a2538",
              command=batch_delete).pack(side="right", padx=(0, 8))
    tk.Button(btn_frame, text="📁 文件所在位置", bg="#23234a", fg="#e8e8f0",
              font=(FONT, 10), relief="flat", bd=0, padx=14, pady=6,
              cursor="hand2", activebackground="#2e2e5e",
              command=open_folder).pack(side="right")

    # ---- 网格区 ----
    grid_frame = tk.Frame(root, bg=BG)
    grid_frame.pack(fill="both", expand=True, padx=16, pady=(4, 4))

    def delete_files(paths):
        """删除文件（带确认），失败文件跳过并提示"""
        if not paths:
            return
        if not messagebox.askyesno("确认删除",
                                   f"确定删除选中的 {len(paths)} 个文件吗？\n删除后不可恢复！"):
            return
        ok = 0
        for p in paths:
            try:
                os.remove(p)
                ok += 1
            except Exception:
                pass
        rebuild_grid()
        messagebox.showinfo("删除完成", f"已删除 {ok} 个文件")

    def show_preview(path):
        """单图检视窗口：大图 + 复制 / 删除 / 关闭"""
        win = tk.Toplevel(root)
        win.title(os.path.basename(path))
        win.configure(bg=BG)
        win.transient(root)
        win.resizable(False, False)

        try:
            img = Image.open(path)
            img.thumbnail((480, 480))
        except Exception as e:
            messagebox.showerror("打开失败", str(e))
            win.destroy()
            return
        photo = ImageTk.PhotoImage(img)
        win._photo_ref = photo  # 防 GC

        tk.Label(win, image=photo, bg=BG).pack(padx=16, pady=(16, 10))

        btns = tk.Frame(win, bg=BG)
        btns.pack(pady=(0, 14))

        def copy_img():
            if copy_image_to_clipboard(path):
                messagebox.showinfo("已复制",
                                    "图片已复制到剪贴板，可直接粘贴到聊天窗口、画图等任何应用")
            else:
                messagebox.showerror("复制失败", "无法复制到剪贴板")

        def del_img():
            win.destroy()
            delete_files([path])

        tk.Button(btns, text="📋 复制图片", bg="#23234a", fg="#e8e8f0",
                  font=(FONT, 10), relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2", activebackground="#2e2e5e",
                  command=copy_img).pack(side="left", padx=6)
        tk.Button(btns, text="🗑 删除", bg="#3a1f2e", fg=RED,
                  font=(FONT, 10), relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2", activebackground="#4a2538",
                  command=del_img).pack(side="left", padx=6)
        tk.Button(btns, text="关闭", bg="#23234a", fg=DIM,
                  font=(FONT, 10), relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2", activebackground="#2e2e5e",
                  command=win.destroy).pack(side="left", padx=6)

    def rebuild_grid():
        """重建网格 + 更新信息栏（保留当前页，越界时回退）"""
        for w in grid_frame.winfo_children():
            w.destroy()
        thumb_refs.clear()
        page_paths.clear()
        check_vars.clear()

        imgs = list_images()
        total = len(imgs)
        if not imgs:
            info_var.set("暂无已生成的图片")
            page_var.set("第 0 / 0 页")
            return

        max_page = max(0, (total - 1) // PAGE_SIZE)
        state["page"] = min(state["page"], max_page)
        page = state["page"]

        page_imgs = imgs[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        page_paths.extend(page_imgs)

        cols = 4
        for i, path in enumerate(page_imgs):
            r, c = divmod(i, cols)
            cell = tk.Frame(grid_frame, bg=PANEL, padx=6, pady=6)
            cell.grid(row=r, column=c, padx=5, pady=5, sticky="n")

            # 缩略图按钮（点击检视）
            photo = None
            try:
                img = Image.open(path)
                img.thumbnail((THUMB, THUMB))
                photo = ImageTk.PhotoImage(img)
                thumb_refs.append(photo)
            except Exception:
                photo = None
            if photo:
                btn = tk.Button(cell, image=photo, bg=PANEL, relief="flat",
                                bd=0, cursor="hand2",
                                command=lambda p=path: show_preview(p))
                btn.image = photo
                btn.pack()
            else:
                tk.Label(cell, text="无法预览", bg=PANEL, fg=DIM,
                         font=(FONT, 8), width=THUMB // 9,
                         height=THUMB // 18).pack()

            # 文件名（截断显示）
            name = os.path.basename(path)
            tk.Label(cell, text=name[:16] + ("…" if len(name) > 16 else ""),
                     bg=PANEL, fg=DIM, font=(FONT, 8)).pack()

            # 勾选（批量删除）
            var = tk.BooleanVar(value=False)
            check_vars.append(var)
            tk.Checkbutton(cell, text="选择", variable=var, bg=PANEL,
                           fg=DIM, selectcolor="#1d1d42", font=(FONT, 8),
                           activebackground=PANEL,
                           activeforeground=DIM).pack()

        info_var.set(f"共 {total} 张 · 第 {page + 1}/{max_page + 1} 页")
        page_var.set(f"第 {page + 1} / {max_page + 1} 页")

    def prev_page():
        if state["page"] > 0:
            state["page"] -= 1
            rebuild_grid()

    def next_page():
        imgs = list_images()
        max_page = max(0, (len(imgs) - 1) // PAGE_SIZE)
        if state["page"] < max_page:
            state["page"] += 1
            rebuild_grid()

    # ---- 翻页区 ----
    page_bar = tk.Frame(root, bg=BG)
    page_bar.pack(side="bottom", fill="x", pady=(0, 14))

    page_var = tk.StringVar(value="")

    tk.Button(page_bar, text="◀ 上一页", bg="#23234a", fg="#e8e8f0",
              font=(FONT, 10), relief="flat", bd=0, padx=18, pady=6,
              cursor="hand2", activebackground="#2e2e5e",
              command=prev_page).pack(side="left", padx=20)

    tk.Label(page_bar, textvariable=page_var, bg=BG, fg=DIM,
             font=(FONT, 10)).pack(side="left")

    tk.Button(page_bar, text="下一页 ▶", bg="#23234a", fg="#e8e8f0",
              font=(FONT, 10), relief="flat", bd=0, padx=18, pady=6,
              cursor="hand2", activebackground="#2e2e5e",
              command=next_page).pack(side="left", padx=20)

    # 首次构建
    rebuild_grid()


# ---------- 页面5：设置页（分辨率 / AI 调用 / 制图需求） ----------
# 常用分辨率选项（值 = 配置文件里的 "宽x高" 字符串）
RESOLUTION_OPTIONS = [
    ("🌐 自动（根据屏幕）", None),
    ("📱 小窗口 480 × 600", "480x600"),
    ("💻 标准 580 × 700", "580x700"),
    ("🖥️ 大窗口 640 × 800", "640x800"),
    ("🖥️ 超大窗口 720 × 900", "720x900"),
]

# 默认值（保存/加载缺失字段时使用）
DEFAULTS = {
    "api_url": URL,
    "api_key": API_KEY,
    "model": "Kwai-Kolors/Kolors",
    "image_size": "1024x1024",
    "num_inference_steps": 30,
    "guidance_scale": 5,
    "rate_per_minute": 12,
}


def show_settings_page(root):
    """设置页：三个 Tab（分辨率 / AI 调用 / 制图需求），参考 Windows 属性面板的标签-值布局"""
    import tkinter as tk
    from tkinter import ttk, messagebox

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    GREEN = "#5cd68b"
    RED = "#ff6b6b"

    root.configure(bg=BG)
    apply_window_size(root, 620, 520)

    def go_back():
        clear_window(root)
        build_launch_page(root)

    build_top_bar(root, "⚙ 设置", right_btn=("← 返回", go_back))

    # ---- 标题 ----
    tk.Label(root, text="设 置", bg=BG, fg=GOLD,
             font=(FONT, 16, "bold")).pack(pady=(14, 4))
    tk.Label(root, text="修改后请点击「保存并应用」使所有设置生效", bg=BG, fg=DIM,
             font=(FONT, 9)).pack()

    # ---- ttk.Notebook 三 Tab ----
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab",
                    background=PANEL, foreground=DIM,
                    padding=(18, 6), font=(FONT, 10))
    style.map("TNotebook.Tab",
              background=[("selected", GOLD)],
              foreground=[("selected", "#14142b")])
    # 模型历史下拉框（Combobox）暗色样式
    style.configure("Dark.TCombobox",
                    fieldbackground="#1d1d42", background="#1d1d42",
                    foreground="#e8e8f0", arrowcolor=GOLD,
                    bordercolor="#3b3b6e", lightcolor="#3b3b6e",
                    darkcolor="#3b3b6e", selectbackground="#2a2a55",
                    selectforeground=GOLD, padding=4)
    style.map("Dark.TCombobox",
              fieldbackground=[("readonly", "#1d1d42")],
              foreground=[("readonly", "#e8e8f0")])

    notebook = ttk.Notebook(root, style="TNotebook")
    notebook.pack(fill="both", expand=True, padx=40, pady=(10, 8))

    cfg = load_settings()
    fields = {}   # 字段名 → tk.StringVar（保存时统一读取）

    # ===== 公共工具：标签 + 值两列网格 =====
    def make_label_value_grid(parent):
        """返回一个可放 (label, widget) 行的 Frame，行号依次自增"""
        frame = tk.Frame(parent, bg=PANEL, padx=20, pady=14)
        frame.columnconfigure(1, weight=1)
        row = {"i": 0}

        def add(label_text, widget):
            tk.Label(frame, text=label_text, bg=PANEL, fg="#e8e8f0",
                     font=(FONT, 10), anchor="w").grid(
                row=row["i"], column=0, sticky="w", padx=(0, 14), pady=6)
            widget.grid(row=row["i"], column=1, sticky="ew", pady=6)
            row["i"] += 1
        return frame, add

    def make_entry(parent, var, show=None):
        """创建 Entry，返回但不调用 pack（由 add 函数 grid 到 frame）"""
        e = tk.Entry(parent, textvariable=var, bg="#1d1d42",
                     fg="#e8e8f0", insertbackground=GOLD,
                     font=(FONT, 10), relief="flat", bd=0,
                     highlightthickness=1, highlightbackground="#3b3b6e",
                     highlightcolor=GOLD, show=show)
        return e

    # ===== Tab1：分辨率 =====
    tab_res = tk.Frame(notebook, bg=PANEL)
    notebook.add(tab_res, text="📐 分辨率")

    res_frame, res_add = make_label_value_grid(tab_res)
    res_frame.pack(fill="both", expand=True, padx=8, pady=8)

    res_var = tk.StringVar(value=cfg.get("window_size") or "auto")
    fields["window_size"] = res_var

    res_combo = tk.OptionMenu(res_frame, res_var, "auto",
                              *[v for _, v in RESOLUTION_OPTIONS if v])
    res_combo.configure(bg="#1d1d42", fg="#e8e8f0", font=(FONT, 10),
                         highlightthickness=0, relief="flat", bd=0,
                         activebackground="#2a2a55")
    res_combo["menu"].configure(bg="#1d1d42", fg="#e8e8f0", font=(FONT, 9))
    res_add("窗口分辨率", res_combo)

    tk.Label(tab_res,
             text="💡 「自动」会按屏幕尺寸自适应；指定分辨率会按该尺寸（屏幕不够时自动压缩）",
             bg=PANEL, fg=GOLD, font=(FONT, 9, "bold"),
             wraplength=480, justify="left").pack(anchor="w", padx=28, pady=(0, 8))

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    tk.Label(tab_res, text=f"🖥 当前屏幕：{sw} × {sh}", bg=PANEL, fg=GOLD,
             font=(FONT, 9)).pack(anchor="w", padx=28)

    # ===== Tab2：AI 调用 =====
    tab_ai = tk.Frame(notebook, bg=PANEL)
    notebook.add(tab_ai, text="🤖 AI 调用")

    ai_frame, ai_add = make_label_value_grid(tab_ai)
    ai_frame.pack(fill="both", expand=True, padx=8, pady=8)

    # ---- 平台选择（从 providers 注册表动态生成，新增平台自动出现）----
    provider_var = tk.StringVar(value=cfg.get("provider", "siliconflow"))
    fields["provider"] = provider_var
    # 用显示名作为下拉项，保存时映射回 id
    provider_display = {
        pid: providers.get_provider_display_name(pid)
        for pid in providers.get_provider_ids()
    }
    current_display = provider_display.get(provider_var.get(), provider_var.get())
    provider_menu = tk.OptionMenu(ai_frame, provider_var, *provider_display.values())
    provider_menu.configure(bg="#1d1d42", fg="#e8e8f0", font=(FONT, 10),
                            highlightthickness=0, relief="flat", bd=0,
                            activebackground="#2a2a55")
    provider_menu["menu"].configure(bg="#1d1d42", fg="#e8e8f0", font=(FONT, 9))
    # 当前值（id）显示为对应显示名
    provider_var.set(current_display)
    ai_add("平台", provider_menu)

    # 平台指引文字（随选择动态更新：选自定义平台时给出填写说明）
    provider_hint = tk.Label(tab_ai, text="", bg=PANEL, fg=GOLD,
                             font=(FONT, 8), wraplength=480, justify="left")
    provider_hint.pack(anchor="w", padx=28, pady=(0, 8))

    def on_provider_change(*_):
        sel = provider_var.get()
        # 显示名 → id
        pid = next((k for k, v in provider_display.items() if v == sel), sel)
        hint = getattr(providers, "PLATFORM_HINTS", {}).get(pid, "")
        provider_hint.config(text=hint)
    provider_var.trace_add("write", on_provider_change)
    on_provider_change()   # 初始化显示

    # 默认状态下 URL / Key / 模型均为空，由用户填写（隐私 + 不预填）
    url_var = tk.StringVar(value=cfg.get("api_url", ""))
    fields["api_url"] = url_var
    ai_add("API URL", make_entry(ai_frame, url_var))

    # 选「硅基流动」自动填好默认 URL（用户无需手抄）；其他平台留空由用户填写
    def auto_fill_url(*_):
        sel = provider_var.get()
        pid = next((k for k, v in provider_display.items() if v == sel), sel)
        if pid == "siliconflow" and not url_var.get().strip():
            cls = providers.PROVIDER_REGISTRY.get("siliconflow")
            if cls and getattr(cls, "default_url", ""):
                url_var.set(cls.default_url)
    provider_var.trace_add("write", auto_fill_url)
    auto_fill_url()   # 初始化时也检查（首次打开设置页）

    key_var = tk.StringVar(value=cfg.get("api_key", ""))
    fields["api_key"] = key_var
    ai_add("API Key", make_entry(ai_frame, key_var, show="•"))

    # 模型名称：可编辑下拉框（Combobox）——可直接输入新模型，也可从「成功使用过的历史」选择
    model_var = tk.StringVar(value=cfg.get("model", ""))
    fields["model"] = model_var

    def _current_provider_id():
        sel = provider_var.get()
        return next((k for k, v in provider_display.items() if v == sel), sel)

    def _refresh_model_history(*_):
        """按当前平台刷新模型历史下拉选项（保留用户已输入的值）"""
        cur = model_var.get()
        model_combo["values"] = get_model_history(_current_provider_id())
        # 刷新时不覆盖用户正在输入的值；若当前值不在历史中，更新显示
        if cur:
            model_var.set(cur)

    model_combo = ttk.Combobox(ai_frame, textvariable=model_var, state="normal",
                               font=(FONT, 10), height=8,
                               style="Dark.TCombobox")   # height=8：下拉最多 8 条，超出自带滚动
    ai_add("模型名称", model_combo)
    # 平台切换时刷新历史选项；初始化也刷新
    provider_var.trace_add("write", _refresh_model_history)
    _refresh_model_history()

    # ---- 参考图/图生图能力声明（D：用户兜底，持久化到 image_input_overrides）----
    # 两个互斥复选框：
    #   「图生图模式」→ 存 true（批量不可用，单张必用参考图）
    #   「参考图兼容」→ 存 "both"（文生图+图生图，批量可用，单张参考图可选）
    img_input_var = tk.BooleanVar(value=False)     # 图生图模式
    compat_var = tk.BooleanVar(value=False)        # 参考图兼容模式

    def _sync_img_input_check():
        """初始值按当前 (provider, model) 解析（D 持久化 > A 静态表 > B 探测）"""
        pid = next((k for k, v in provider_display.items() if v == provider_var.get()),
                   provider_var.get())
        model = model_var.get().strip()
        cap = None
        if model:
            cap = providers.resolve_image_input_capability(
                pid, model, overrides=cfg.get("image_input_overrides", {}))
        if cap == "img":
            img_input_var.set(True); compat_var.set(False)
        elif cap == "both":
            img_input_var.set(False); compat_var.set(True)
        else:
            img_input_var.set(False); compat_var.set(False)

    def _mutual_exclude_img(*_):
        # 勾选「图生图」→ 取消「兼容」
        if img_input_var.get():
            compat_var.set(False)

    def _mutual_exclude_compat(*_):
        # 勾选「兼容」→ 取消「图生图」
        if compat_var.get():
            img_input_var.set(False)

    _sync_img_input_check()
    img_input_var.trace_add("write", _mutual_exclude_img)
    compat_var.trace_add("write", _mutual_exclude_compat)

    img_input_chk = tk.Checkbutton(
        ai_frame, text="图生图模式（仅单张生成，需参考图）",
        variable=img_input_var, bg="#1d1d42", fg="#e8e8f0",
        selectcolor="#2a2a55", font=(FONT, 10),
        activebackground="#1d1d42", activeforeground=GOLD,
        highlightthickness=0, bd=0, cursor="hand2")
    ai_add("参考图输入", img_input_chk)

    compat_chk = tk.Checkbutton(
        ai_frame, text="参考图兼容（文生图 + 图生图，批量可用）",
        variable=compat_var, bg="#1d1d42", fg="#e8e8f0",
        selectcolor="#2a2a55", font=(FONT, 10),
        activebackground="#1d1d42", activeforeground=GOLD,
        highlightthickness=0, bd=0, cursor="hand2")
    ai_add("参考图兼容", compat_chk)

    # 模型名/平台变化时联动更新复选框（D/A/B 解析）
    model_var.trace_add("write", lambda *_: _sync_img_input_check())
    provider_var.trace_add("write", lambda *_: _sync_img_input_check())

    tk.Label(tab_ai, text="⚠ 密钥以明文保存在本地 settings.json，请勿分享配置文件",
             bg=PANEL, fg=RED, font=(FONT, 8)).pack(anchor="w", padx=28, pady=(0, 8))

    def test_api():
        """测试当前输入的 API 配置是否可用（先选方式：完整测试 / 仅格式检查）"""
        url = url_var.get().strip()
        key = key_var.get().strip()
        model = model_var.get().strip()
        if not url:
            messagebox.showwarning("提示", "请先填写 API URL")
            return
        if not key:
            messagebox.showwarning("提示", "请先填写 API Key")
            return
        # 先弹窗选择测试方式
        mode = ask_test_mode(root)
        if mode is None:      # 取消
            return

        # ===== ② 仅格式检查（免费，不发请求）=====
        if mode == "format":
            issues = []
            if not url.startswith(("http://", "https://")):
                issues.append("API URL 应以 http:// 或 https:// 开头")
            if " " in url:
                issues.append("API URL 不能包含空格")
            if not key.startswith(("sk-", "key-", "api")):
                issues.append("API Key 通常以 sk- 开头（请核对是否复制完整）")
            if not model:
                issues.append("模型名称不能为空")
            if issues:
                messagebox.showwarning(
                    "格式检查未通过",
                    "以下项目可能有问题：\n" + "\n".join(f"· {i}" for i in issues) +
                    "\n\n（仅做格式检查，未向平台发送请求，未消耗任何额度）")
            else:
                messagebox.showinfo(
                    "格式检查通过",
                    "URL / Key / 模型名格式均正常。\n"
                    "（仅做格式检查，未向平台发送请求，未消耗任何额度）\n"
                    "如需验证 Key 真实有效，请使用「完整测试」。")
            return

        # ===== ① 完整测试（真实请求，消耗少量额度）=====
        # 用临时配置构造 provider 实例（不落盘）
        tmp_cfg = dict(cfg)
        tmp_cfg["api_url"] = url
        tmp_cfg["api_key"] = key
        tmp_cfg["model"] = model
        tmp_cfg["provider"] = provider_var.get()
        prov = providers.create_provider(tmp_cfg)
        sel_display = provider_var.get()
        pid = next((k for k, v in provider_display.items() if v == sel_display), sel_display)
        # 图生图/兼容模型需要 image 字段 → 测连接时生成 1x1 占位图
        test_ref = None
        cap = providers.resolve_image_input_capability(
            pid, model, overrides=load_settings().get("image_input_overrides", {}))
        if cap in ("img", "both"):
            try:
                from io import BytesIO
                from PIL import Image
                import tempfile as _tf
                buf = BytesIO()
                Image.new('RGB', (1, 1), (0, 0, 0)).save(buf, 'PNG')
                test_ref = os.path.join(_tf.gettempdir(), '_weaver_test_ref.png')
                with open(test_ref, 'wb') as f:
                    f.write(buf.getvalue())
            except Exception:
                test_ref = None
        try:
            ok, msg = prov.test_connection(reference_image=test_ref)
        finally:
            # 清理临时占位图
            if test_ref and os.path.exists(test_ref):
                try:
                    os.remove(test_ref)
                except Exception:
                    pass
        if ok:
            # B 探测：模型列表接口若返回能力信息 → 自动勾选/取消复选框
            try:
                b = prov.probe_model_capability(model)
                if b is not None:
                    _bn = providers._normalize_cap(b)
                    if _bn == "img":
                        img_input_var.set(True); compat_var.set(False)
                        msg += "\n已自动识别该模型支持图生图（仅单张）"
                    elif _bn == "both":
                        img_input_var.set(False); compat_var.set(True)
                        msg += "\n已自动识别该模型支持图生图兼容（文生图+图生图）"
                    else:
                        img_input_var.set(False); compat_var.set(False)
                        msg += "\n该模型为文生图，不支持参考图输入"
            except Exception:
                pass
            messagebox.showinfo("测试成功", msg)
        else:
            messagebox.showwarning("测试失败", msg)

    tk.Button(tab_ai, text="🔌 测试连接", bg="#23234a", fg="#e8e8f0",
              font=(FONT, 10), relief="flat", bd=0,
              padx=18, pady=6, cursor="hand2",
              activebackground="#2e2e5e",
              command=test_api).pack(anchor="e", padx=28, pady=(0, 8))

    # ===== Tab3：制图需求 =====
    tab_draw = tk.Frame(notebook, bg=PANEL)
    notebook.add(tab_draw, text="🎨 制图需求")

    draw_frame, draw_add = make_label_value_grid(tab_draw)
    draw_frame.pack(fill="both", expand=True, padx=8, pady=8)

    # image_size 常见预设 + 自定义
    size_var = tk.StringVar(value=cfg.get("image_size", DEFAULTS["image_size"]))
    fields["image_size"] = size_var
    draw_add("图片尺寸", make_entry(draw_frame, size_var))

    steps_var = tk.StringVar(value=str(cfg.get("num_inference_steps", DEFAULTS["num_inference_steps"])))
    fields["num_inference_steps"] = steps_var
    draw_add("AI 思考步数", make_entry(draw_frame, steps_var))

    guide_var = tk.StringVar(value=str(cfg.get("guidance_scale", DEFAULTS["guidance_scale"])))
    fields["guidance_scale"] = guide_var
    draw_add("提示词遵守度", make_entry(draw_frame, guide_var))

    rate_var = tk.StringVar(value=str(cfg.get("rate_per_minute", DEFAULTS["rate_per_minute"])))
    fields["rate_per_minute"] = rate_var
    draw_add("每分钟生成张数", make_entry(draw_frame, rate_var))

    tk.Label(tab_draw,
             text="💡 步数越多越慢但质量更好；提示词遵守度越高越贴近输入（过高易失真）；"
                  "每分钟张数限制间隔，防止 API 限流",
             bg=PANEL, fg=GOLD, font=(FONT, 9, "bold"),
             wraplength=480, justify="left").pack(anchor="w", padx=28, pady=(0, 4))

    # ---- 操作按钮 ----
    op_frame = tk.Frame(root, bg=BG)
    op_frame.pack(pady=(0, 6))

    # 版本号提示（底部居中，弱化显示）
    tk.Label(root, text=f"编织者 v{VERSION}", bg=BG, fg="#55557a",
             font=(FONT, 8)).pack(pady=(0, 8))

    def on_save():
        # 收集所有字段，校验后写入
        new_cfg = load_settings()
        # 分辨率
        ws = res_var.get()
        if ws == "auto":
            new_cfg.pop("window_size", None)
        else:
            new_cfg["window_size"] = ws
        # AI 调用（空则保存空，不偷偷塞默认值——由用户自己填写）
        new_cfg["api_url"]   = url_var.get().strip()
        new_cfg["api_key"]   = key_var.get().strip()
        new_cfg["model"]     = model_var.get().strip()
        # 平台：下拉显示名 → 映射回 id（未知值按原样保存）
        sel_display = provider_var.get()
        pid = next((k for k, v in provider_display.items() if v == sel_display), sel_display)
        new_cfg["provider"] = pid
        # 参考图能力声明（D 持久化）："{provider}::{model}" → true（图生图）/ "both"（兼容）/ false（文生图）
        model = model_var.get().strip()
        if model:
            ov = new_cfg.get("image_input_overrides") or {}
            if not isinstance(ov, dict):
                ov = {}
            if compat_var.get():
                ov[f"{pid}::{model}"] = "both"
            else:
                ov[f"{pid}::{model}"] = bool(img_input_var.get())
            new_cfg["image_input_overrides"] = ov
        # 制图需求（校验数字）
        try:
            steps = int(steps_var.get())
            guide = float(guide_var.get())
            rate = float(rate_var.get())
            if steps < 1 or steps > 100: raise ValueError("steps 范围 1-100")
            if guide < 0 or guide > 30:   raise ValueError("guide 范围 0-30")
            if rate < 0.5 or rate > 120:   raise ValueError("rate 范围 0.5-120")
        except ValueError as e:
            messagebox.showerror("参数错误", f"制图需求数值不合法：\n{e}")
            return
        size = size_var.get().strip()
        # image_size 简单校验
        if not re.match(r'^\d{2,4}x\d{2,4}$', size):
            messagebox.showerror("参数错误", "图片尺寸格式应为「宽x高」，如 1024x1024")
            return
        new_cfg["image_size"] = size
        new_cfg["num_inference_steps"] = steps
        new_cfg["guidance_scale"] = guide
        new_cfg["rate_per_minute"] = rate

        save_settings(new_cfg)
        apply_window_size(root, 620, 520)   # 分辨率立即生效
        messagebox.showinfo("已保存", "所有设置已保存并立即生效。\n下次启动会自动应用。")

    tk.Button(op_frame, text="✅ 保存并应用", bg=GOLD, fg="#14142b",
              font=(FONT, 12, "bold"), relief="flat", bd=0,
              padx=26, pady=10, cursor="hand2",
              activebackground="#ffd98a", activeforeground="#14142b",
              command=on_save).pack(side="left", padx=8)

    tk.Button(op_frame, text="← 返回", bg="#23234a", fg=DIM,
              font=(FONT, 11), relief="flat", bd=0,
              padx=22, pady=10, cursor="hand2",
              activebackground="#2e2e5e", activeforeground="#e8e8f0",
              command=go_back).pack(side="left", padx=8)


# ---------- 页面6：单个生成页（手动输入提示词） ----------
def show_single_page(root):
    """单个生成：对话框输入提示词，点击生成单张图片"""
    import tkinter as tk
    from tkinter import messagebox
    import threading

    FONT = FONT_NAME
    BG = "#0d0d1a"
    PANEL = "#15152e"
    GOLD = "#f0c75e"
    DIM = "#8a8aa8"
    GREEN = "#5cd68b"
    RED = "#ff6b6b"

    root.configure(bg=BG)
    apply_window_size(root, 580, 480)

    def go_back():
        clear_window(root)
        show_mode_page(root)

    build_top_bar(root, "🖼️ 单个生成", right_btn=("← 返回", go_back))

    # ---- 标题 ----
    tk.Label(root, text="单 个 生 成", bg=BG, fg=GOLD,
             font=(FONT, 18, "bold")).pack(pady=(24, 4))
    tk.Label(root, text="输入提示词，生成一张专属图片", bg=BG, fg=DIM,
             font=(FONT, 10)).pack()

    # ---- 提示词输入框 ----
    panel = tk.Frame(root, bg=PANEL, padx=24, pady=14)
    panel.pack(fill="x", padx=40, pady=(20, 10))

    tk.Label(panel, text="提示词（Prompt）", bg=PANEL, fg="#e8e8f0",
             font=(FONT, 11, "bold")).pack(anchor="w", pady=(0, 6))

    prompt_var = tk.StringVar()
    prompt_entry = tk.Entry(panel, textvariable=prompt_var, bg="#1d1d42",
                            fg="#e8e8f0", insertbackground=GOLD,
                            font=(FONT, 11), relief="flat", bd=0,
                            highlightthickness=1, highlightbackground="#3b3b6e",
                            highlightcolor=GOLD)
    prompt_entry.pack(fill="x", ipady=8)

    tk.Label(panel,
             text="💡 例如：一只戴着皇冠的橘猫，奇幻风格",
             bg=PANEL, fg=GOLD, font=(FONT, 9, "bold")).pack(anchor="w", pady=(6, 0))

    # ---- 画风输入 ----
    tk.Label(panel, text="画风（可选，默认：卡牌插画·奇幻风格）",
             bg=PANEL, fg="#e8e8f0", font=(FONT, 10), anchor="w").pack(anchor="w", pady=(10, 4))
    style_var = tk.StringVar(value="")
    style_entry = tk.Entry(panel, textvariable=style_var, bg="#1d1d42",
                           fg="#e8e8f0", insertbackground=GOLD,
                           font=(FONT, 11), relief="flat", bd=0,
                           highlightthickness=1, highlightbackground="#3b3b6e",
                           highlightcolor=GOLD)
    style_entry.pack(fill="x", ipady=7)
    tk.Label(panel, text="💡 例如：水墨画风 / 赛博朋克 / 宫崎骏动画风",
             bg=PANEL, fg=GOLD, font=(FONT, 8)).pack(anchor="w", pady=(4, 0))

    # ---- 参考图输入（可选，放在页面内；用户可随时选/清除）----
    from tkinter import filedialog as _filedialog
    ref_section_label = tk.Label(panel, text="参考图（可选，仅图生图/兼容模式生效）",
                                 bg=PANEL, fg="#e8e8f0", font=(FONT, 10), anchor="w")
    ref_section_label.pack(anchor="w", pady=(10, 4))
    reference_image_var = tk.StringVar(value="")

    ref_row = tk.Frame(panel, bg=PANEL)
    ref_row.pack(fill="x")

    def choose_reference():
        p = _filedialog.askopenfilename(
            title="选择参考图",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp"), ("所有文件", "*.*")])
        if p:
            reference_image_var.set(p)
            ref_path_label.config(text=f"✅ {os.path.basename(p)}", fg=GREEN)
            ref_clear_btn.config(state="normal")

    ref_choose_btn = tk.Button(ref_row, text="📁 选择参考图", bg="#23234a", fg="#e8e8f0",
                               font=(FONT, 10), relief="flat", bd=0,
                               padx=14, pady=5, cursor="hand2",
                               activebackground="#2e2e5e", activeforeground=GOLD,
                               command=choose_reference)
    ref_choose_btn.pack(side="left")

    ref_path_label = tk.Label(ref_row, text="未选择", bg=PANEL, fg="#7a7aa0",
                              font=(FONT, 9), wraplength=360, justify="left")
    ref_path_label.pack(side="left", padx=(10, 0), fill="x", expand=True)

    def clear_reference():
        reference_image_var.set("")
        ref_path_label.config(text="未选择", fg="#7a7aa0")
        ref_clear_btn.config(state="disabled")

    ref_clear_btn = tk.Button(ref_row, text="✕ 清除", bg="#23234a", fg=DIM,
                              font=(FONT, 9), relief="flat", bd=0,
                              padx=10, pady=5, cursor="hand2",
                              activebackground="#2e2e5e", activeforeground="#e8e8f0",
                              state="disabled", command=clear_reference)
    ref_clear_btn.pack(side="right", padx=(8, 0))

    ref_hint_label = tk.Label(panel, text="💡 不选也行；纯文生图模式（text）会自动忽略，img/both 模式会真正使用",
                              bg=PANEL, fg=GOLD, font=(FONT, 8))
    ref_hint_label.pack(anchor="w", pady=(4, 0))

    # ---- 按模型能力隐藏参考图区：仅 img/both（支持参考图）才显示，text 模式彻底隐藏 ----
    _cap_single = providers.resolve_image_input_capability(
        get_api_settings()["provider"], get_api_settings()["model"],
        overrides=load_settings().get("image_input_overrides", {}))
    if _cap_single not in ("img", "both"):
        ref_section_label.pack_forget()
        ref_row.pack_forget()
        ref_hint_label.pack_forget()

    # ---- 生成参数提示（实时读取设置） ----
    _api_s = get_api_settings()
    tk.Label(root, text=f"⚙️ 画幅 {_api_s['image_size']} · 模型 {_api_s['model']}",
             bg=BG, fg=DIM, font=(FONT, 9)).pack(pady=(0, 2))

    # ---- 状态 ----
    status_label = tk.Label(root, text="", bg=BG, fg=DIM, font=(FONT, 10))
    status_label.pack(pady=(8, 2))

    # ---- 按钮区 ----
    op_frame = tk.Frame(root, bg=BG)
    op_frame.pack(pady=(12, 0))

    def on_generate():
        """生成单张图片（后台线程，UI 不卡顿）"""
        prompt = prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("提示", "请先输入提示词")
            return
        if not any('\u4e00' <= ch <= '\u9fff' for ch in prompt):
            if not messagebox.askyesno("确认", "提示词中没有中文，仍要继续吗？"):
                return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 文件名：取提示词前几个字 + 时间戳，避免重复覆盖
        stamp = time.strftime("%H%M%S")
        safe = re.sub(r'[\\/:*?"<>|\s]+', '_', prompt).strip('._')[:30] or "single"
        save_path = get_unique_save_path(
            os.path.join(OUTPUT_DIR, f"{safe}_{stamp}.png"))

        # ---- 参考图（直接读页面状态，用户自行选择）----
        reference_image = reference_image_var.get().strip() or None

        # ---- 图生图模式强制参考图：仅图生图（img）必须提供，兼容（both）可选 ----
        _api_s = get_api_settings()
        _cap = providers.resolve_image_input_capability(
            _api_s["provider"], _api_s["model"],
            overrides=load_settings().get("image_input_overrides", {}))
        if _cap == "img" and not reference_image:
            messagebox.showwarning(
                "需要参考图",
                f"当前模型「{_api_s['model']}」为图生图模式，必须选择一张参考图才能生成。\n"
                f"请点击上方「📁 选择参考图」上传图片。")
            return

        generate_btn.config(state="disabled")
        status_label.config(text="⏳ 正在生成…请稍候", fg="#e8e8f0")

        def worker():
            # 画风：用户自定义或默认
            user_style = style_var.get().strip()
            style_suffix = user_style if user_style else "奇幻卡牌插画，精美"
            # img（仅图生图/编辑模型）不拼画风前缀——编辑场景下多余
            final_prompt = prompt if _cap == "img" else f"{style_suffix}，{prompt}"
            ok, err = generate_image(final_prompt, save_path, reference_image)
            if ok:
                # 记录成功使用的模型到历史（供设置页下拉选择）
                _a = get_api_settings()
                record_model_usage(_a["provider"], _a["model"])
                status_label.config(
                    text=f"✅ 已保存：{os.path.basename(save_path)} → {OUTPUT_DIR}/",
                    fg=GREEN)
            else:
                status_label.config(text=f"❌ 生成失败：{err}", fg=RED)
                # 详细原因弹窗（仅展示前几行，避免弹窗过高）
                err_short = err if len(err) <= 300 else err[:300] + "..."
                messagebox.showerror(
                    "生成失败",
                    f"提示词：{prompt}\n\n失败原因：\n{err_short}\n\n提示：常见原因是 API 余额不足、网络不稳定或提示词违规。")
            generate_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    generate_btn = tk.Button(op_frame, text="⚡ 生 成 图 片", bg=GOLD, fg="#14142b",
                             font=(FONT, 12, "bold"), relief="flat", bd=0,
                             padx=28, pady=10, cursor="hand2",
                             activebackground="#ffd98a", activeforeground="#14142b",
                             command=on_generate)
    generate_btn.pack(side="left", padx=8)

    tk.Button(op_frame, text="← 返回", bg="#23234a", fg=DIM,
              font=(FONT, 11), relief="flat", bd=0,
              padx=22, pady=10, cursor="hand2",
              activebackground="#2e2e5e", activeforeground="#e8e8f0",
              command=go_back).pack(side="left", padx=8)

    # 回车键快速生成
    prompt_entry.bind("<Return>", lambda e: on_generate())
    prompt_entry.focus_set()


# ============= 主程序 =============
def main():
    """唯一入口：启动 GUI（noconsole 打包下不会因 input()/print() 崩溃）"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    show_launch_window()


if __name__ == "__main__":
    main()
