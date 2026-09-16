# =============================================================================
# Provider 适配器层（Provider Pattern）
# -----------------------------------------------------------------------------
# 结构：抽象基类 → OpenAI 兼容通用实现 → 各平台（只覆写元数据）
#       + 参考范本（演示不同字段协议的平台如何接入）
#
# 新增平台步骤（只需在 providers.py 内操作）：
#   1. 若目标平台是「OpenAI 兼容 /v1/images/generations」→ 继承 OpenAICompatibleProvider，
#      只覆写 id / display_name / default_url / default_model；
#   2. 若字段协议不同（如 DALL·E、Stability）→ 继承 ProviderBase，实现 generate() 与
#      test_connection()（可参考下方 OpenAIProvider / StabilityProvider 范本）；
#   3. 在 PROVIDER_REGISTRY 注册；设置页下拉框自动出现。
#
# 配置键（settings.json）：
#   api_url / api_key / model / image_size / num_inference_steps / guidance_scale
#   provider（平台 id）
#   response_format（可选：""/url/b64_json，请求时协商返回格式）
#   extra_headers（可选：dict，附加/覆盖请求头，兼容 api-key 等非 Bearer 鉴权）
#   image_input_overrides（可选：dict，键 "{provider_id}::{model_id}" → true/false，
#                          持久化用户对「该模型是否支持参考图/图生图」的手动声明）
# =============================================================================

import base64
import io
import os
import sys
import requests
from PIL import Image


# =============================================================================
# A：模型级静态能力表（零网络请求）
# -----------------------------------------------------------------------------
# 能力取决于 (provider, model) 组合，而非平台整体。键 = (provider_id, model_id)。
# 值（三态）：
#   False → 仅文生图；True → 仅图生图；"both" → 文生图 + 图生图兼容。
# 已知的模型直接查表；未收录的返回 None（交给用户声明 D 兜底）。
# 注意：此表只是默认值，用户在设置页勾选的持久化结果（D）优先级更高，可覆盖。
# =============================================================================
KNOWN_IMAGE_INPUT = {
    # Kolors 支持文生图 + 图生图（参考图兼容）——已核实，勿改回 False 否则新用户图生图功能被藏起来
    ("siliconflow", "Kwai-Kolors/Kolors"): "both",
    # 通义万相 Z-Image 系列：仅文生图（不支持参考图输入）
    ("siliconflow", "Tongyi-MAI/Z-Image-Turbo"): False,
    ("siliconflow", "Tongyi-MAI/Z-Image"): False,
    # 通义万相 Qwen-Image-Edit：仅图生图（编辑模型，必须提供参考图）
    ("siliconflow", "Qwen/Qwen-Image-Edit-2509"): True,
    # 以后确认支持的模型在此加：
    # ("siliconflow", "Qwen/Qwen-Image-Edit"): True,
}


def _load_image_overrides():
    """从 settings.json 读取用户持久化的能力声明（D）。缺字段/文件不存在 → {}（向后兼容）"""
    try:
        if getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, "settings.json"), 'r', encoding='utf-8') as f:
            import json
            ov = json.load(f).get("image_input_overrides") or {}
            return ov if isinstance(ov, dict) else {}
    except Exception:
        return {}


def _normalize_cap(value):
    """把能力值统一为三态：'text'（仅文生图）/ 'img'（仅图生图）/ 'both'（兼容）。
    兼容旧数据：True → 'img'，False → 'text'。"""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ("both", "compat", "mixed"):
            return "both"
        if v in ("true", "1", "yes", "img", "image", "img2img"):
            return "img"
        if v in ("false", "0", "no", "text", "txt"):
            return "text"
        return "text"
    return "img" if bool(value) else "text"


def resolve_image_input_capability(provider_id, model_id, overrides=None):
    """统一入口：检测 (provider, model) 的参考图/图生图能力。
    返回三态 'text'（仅文生图）/ 'img'（仅图生图）/ 'both'（文生图+图生图兼容）；
    无法确定时返回 None（表示"让用户决定"）。
    优先级（不发送真实生成请求，符合"禁用 probe"约束）：
      D（用户持久化声明，可覆盖静态表）→ A（静态能力表）→ B（provider 模型列表探测）→ None
    """
    key = f"{provider_id}::{model_id}"
    # D：用户持久化决定（最高优先，跨会话生效；值可为 true/false/"both"）
    ov = overrides if overrides is not None else _load_image_overrides()
    if key in ov:
        return _normalize_cap(ov[key])
    # A：静态能力表
    if (provider_id, model_id) in KNOWN_IMAGE_INPUT:
        return _normalize_cap(KNOWN_IMAGE_INPUT[(provider_id, model_id)])
    # B：provider 模型列表探测（仅当平台真实返回能力信息才可确定）
    cls = PROVIDER_REGISTRY.get(provider_id)
    if cls is not None:
        try:
            inst = cls({"model": model_id})
            b = inst.probe_model_capability(model_id)
            if b is not None:
                return _normalize_cap(b)
        except Exception:
            pass
    return None


class ProviderBase:
    """所有平台适配器的抽象基类：统一接口，子类实现 generate/test_connection"""

    # ---- 子类必须定义 ----
    id = ""               # 存进 settings.json 的 provider 字段值
    display_name = ""     # 设置页下拉框显示名称
    default_url = ""      # 该平台默认 API 地址
    default_model = ""    # 该平台默认模型名

    def __init__(self, cfg):
        # cfg: 从 settings.json 读出的配置字典
        self.url = (cfg.get("api_url") or "").strip() or self.default_url
        self.key = (cfg.get("api_key") or "").strip()
        self.model = (cfg.get("model") or "").strip() or self.default_model
        self.image_size = cfg.get("image_size", "1024x1024")
        self.num_inference_steps = int(cfg.get("num_inference_steps", 30))
        self.guidance_scale = float(cfg.get("guidance_scale", 5))
        # ---- 高级配置（可选，settings.json 手配）----
        self.extra_headers = cfg.get("extra_headers") or {}
        if not isinstance(self.extra_headers, dict):
            self.extra_headers = {}
        self.response_format = (cfg.get("response_format") or "").strip()  # ""/url/b64_json

    def is_configured(self):
        """是否已填写必要的配置（未配置时生成前给出明确提示）"""
        return bool(self.key and self.url)

    # ---- 参考图 / 图生图能力检测（D→A→B 组合，入口在 resolve_image_input_capability）----
    def supports_image_input(self, model_id=None):
        """当前平台 + 模型是否支持参考图输入。返回 True/False；未知返回 None。
        默认实现：查静态能力表 KNOWN_IMAGE_INPUT。子类可覆写（如增加模型列表探测）。"""
        mid = model_id or self.model
        return KNOWN_IMAGE_INPUT.get((self.id, mid))

    def probe_model_capability(self, model_id):
        """B：尝试从「模型列表接口」探测模型能力（零成本、只读）。
        默认不实现（返回 None = 不探测）；子类在平台确实返回能力信息时覆写。"""
        return None

    # ---- 子类实现 ----
    def generate(self, prompt, save_path, reference_image=None):
        """生成一张图片并保存。返回 (ok: bool, err: str)；成功时 err 为空。
        reference_image: 可选，参考图（图片文件路径或二进制 bytes）。
        支持图生图的子类应使用它并自行构造对应请求；默认实现忽略参考图走纯文生图。"""
        raise NotImplementedError

    def test_connection(self, reference_image=None):
        """测试平台连接是否可用。返回 (ok: bool, msg: str)。
        reference_image: 可选参考图（图生图模型必填；设置页测连接时会自动生成 1x1 占位图）。"""
        raise NotImplementedError

    # ---- 通用工具 ----
    def _headers(self):
        """默认 Bearer 鉴权；可被 extra_headers 覆盖（如 api-key 头）"""
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        for k, v in (self.extra_headers or {}).items():
            headers[k] = str(v)
        return headers

    def _post_json(self, payload, timeout=120):
        """POST JSON 并返回 (status, json_dict|None, err)。非 200 或解析失败时 err 非空。"""
        try:
            resp = requests.post(self.url, headers=self._headers(),
                                 json=payload, timeout=timeout)
        except Exception as e:
            return 0, None, f"请求异常: {e}"
        if resp.status_code != 200:
            msg = (resp.text or "")[:200].replace("\n", " ")
            return resp.status_code, None, f"HTTP {resp.status_code}: {msg}"
        try:
            return resp.status_code, resp.json(), ""
        except Exception as e:
            return resp.status_code, None, f"解析响应失败: {e}"

    def _download_and_save(self, image_url, save_path):
        """下载图片二进制并保存为 PNG（抑制 libpng iCCP 警告）"""
        resp = requests.get(image_url, timeout=120)
        img = Image.open(io.BytesIO(resp.content))
        img.save(save_path, "PNG", icc_profile=None, optimize=False)

    def _save_b64(self, b64_str, save_path):
        """base64 字符串解码后保存为 PNG"""
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw))
        img.save(save_path, "PNG", icc_profile=None, optimize=False)

    def _reference_to_b64(self, reference_image, data_uri=True):
        """把参考图（文件路径或 bytes）转成 base64。返回 base64 字符串或 data URI。
        供图生图请求体使用；失败抛异常由调用方捕获。"""
        if isinstance(reference_image, (bytes, bytearray)):
            raw = bytes(reference_image)
        else:
            with open(reference_image, 'rb') as f:
                raw = f.read()
        b64 = base64.b64encode(raw).decode('ascii')
        return f"data:image/png;base64,{b64}" if data_uri else b64


# =============================================================================
# OpenAI 兼容通用实现（POST /v1/images/generations）
#   - 支持 url 与 b64_json 两种返回（b64_json 分支已补）
#   - 请求字段：model / prompt / image_size / batch_size / num_inference_steps / guidance_scale
#   - 可选 response_format 协商
# =============================================================================
class OpenAICompatibleProvider(ProviderBase):
    """OpenAI 兼容文生图接口的通用实现（供硅基流动与自定义平台复用）"""

    def _payload(self, prompt, batch_size=1, minimal=False, reference_image=None):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "image_size": self.image_size,
            "batch_size": batch_size,
            "num_inference_steps": 1 if minimal else self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.response_format:
            payload["response_format"] = self.response_format
        # 参考图（图生图）：OpenAI 兼容端点常见约定为 image 字段（data URI / base64）
        if reference_image is not None:
            payload["image"] = self._reference_to_b64(reference_image)
        return payload

    def _extract_image(self, result):
        """从响应中提取图片数据（健壮性处理）：优先 url，回退 b64_json。
        兼容两种返回容器：OpenAI 风格 data 与 SiliconFlow 官方 images。
        返回 ('url'|'b64', data) 或 (None, err)。"""
        if not isinstance(result, dict):
            return None, "API 返回的不是 JSON 对象"
        # OpenAI 兼容用 data；SiliconFlow 官方接口用 images——两者都兼容
        data = result.get("data") or result.get("images")
        if not isinstance(data, list) or not data:
            return None, "API 返回的 data/images 字段为空或格式不正确"
        item = data[0]
        if not isinstance(item, dict):
            return None, "API 返回的 data/images[0] 格式不正确"
        url = item.get("url")
        if url:
            return ("url", url)
        b64 = item.get("b64_json")
        if b64:
            return ("b64", b64)
        return None, "API 返回数据中没有图片（既无 url 也无 b64_json）"

    def generate(self, prompt, save_path, reference_image=None):
        """POST → 解析 url/b64_json → 下载或解码保存。
        reference_image 非空时按图生图处理（参考图以 base64 进请求体）。"""
        try:
            payload = self._payload(prompt, reference_image=reference_image)
        except Exception as e:
            return False, f"参考图处理失败: {e}"
        status, result, err = self._post_json(payload)
        if err:
            return False, err
        kind, data = self._extract_image(result)
        if not kind:
            return False, data   # data 为错误描述
        try:
            if kind == "url":
                self._download_and_save(data, save_path)
            else:
                self._save_b64(data, save_path)
            return True, ""
        except Exception as e:
            return False, f"下载/保存失败: {e}"

    def test_connection(self, reference_image=None):
        if not self.url:
            return False, "请先填写 API URL"
        if not self.key:
            return False, "请先填写 API Key"
        if not self.model:
            return False, "请先填写模型名称"
        # 最小参数（步数 1、尺寸由平台最小化处理）以节省额度；
        # 图生图模型必填 image 字段，reference_image 由调用方提供（设置页自动生成 1x1 占位）
        status, result, err = self._post_json(
            self._payload("test", batch_size=1, minimal=True, reference_image=reference_image),
            timeout=30)
        if err:
            return False, err
        return True, f"连接成功（HTTP 200）· 模型 {self.model} 可用（测试已消耗少量额度）"


# =============================================================================
# 内置平台：硅基流动 SiliconFlow（OpenAI 兼容，仅覆写元数据）
# =============================================================================
class SiliconFlowProvider(OpenAICompatibleProvider):
    id = "siliconflow"
    display_name = "硅基流动 SiliconFlow"
    default_url = "https://api.siliconflow.cn/v1/images/generations"
    default_model = "Kwai-Kolors/Kolors"
    models_url = "https://api.siliconflow.cn/v1/models"   # 模型列表接口（B 探测用）

    def fetch_models(self):
        """拉取模型列表（GET /v1/models）。返回模型条目列表；失败返回 []。
        仅用于 B 探测：只读、不发生成请求。"""
        if not self.key:
            return []
        try:
            resp = requests.get(self.models_url,
                                headers={"Authorization": f"Bearer {self.key}"},
                                timeout=20)
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = data.get("data") if isinstance(data, dict) else None
            return models if isinstance(models, list) else []
        except Exception:
            return []

    def probe_model_capability(self, model_id):
        """B：从 /v1/models 探测该模型是否带「图生图能力」字段。
        仅当平台返回的能力字段明确存在时才判定；否则返回 None（不猜测）。"""
        try:
            for m in self.fetch_models():
                if not isinstance(m, dict):
                    continue
                if m.get("id") != model_id:
                    continue
                # 常见能力字段：image_input / image_input_support / capabilities.img2img 等
                if "image_input" in m:
                    return bool(m.get("image_input"))
                if "image_input_support" in m:
                    return bool(m.get("image_input_support"))
                caps = m.get("capabilities")
                if isinstance(caps, dict):
                    if "image_input" in caps:
                        return bool(caps.get("image_input"))
                    if "img2img" in caps:
                        return bool(caps.get("img2img"))
                return None   # 模型存在但无能力字段 → 无法确定
        except Exception:
            pass
        return None


# =============================================================================
# 内置平台：自定义平台（OpenAI 兼容 /v1/images/generations）
# -----------------------------------------------------------------------------
# 供「使用其他平台、懂 API 的用户」接入：选此项 → 填 URL / Key / 模型名。
# 仅支持返回 url 或 b64_json 的 OpenAI 兼容文生图接口（PLATFORM_HINTS 已注明）。
# =============================================================================
class CustomProvider(OpenAICompatibleProvider):
    id = "custom"
    display_name = "OpenAI 兼容平台（/v1/images/generations）"
    default_url = ""
    default_model = ""

    def test_connection(self, reference_image=None):
        if not self.url:
            return False, "请先填写 API URL（通常以 /v1/images/generations 结尾）"
        if not self.key:
            return False, "请先填写 API Key"
        if not self.model:
            return False, "请先填写模型名称"
        # 最小参数；图生图模型时由调用方传 reference_image（session 应答也将含 image 字段）
        status, result, err = self._post_json(
            self._payload("test", batch_size=1, minimal=True, reference_image=reference_image),
            timeout=30)
        if err:
            return False, err
        return True, f"连接成功（HTTP 200）· 模型 {self.model} 可用"


# =============================================================================
# 参考范本 1：OpenAI DALL·E 官方接口（字段 size/n，支持 b64_json）
# -----------------------------------------------------------------------------
# 演示「字段协议不同」的平台如何接入：继承 ProviderBase，实现 generate/test_connection。
# 注意：DALL·E 无 num_inference_steps / guidance_scale，字段是 size / n。
# =============================================================================
class OpenAIProvider(ProviderBase):
    id = "openai"
    display_name = "OpenAI DALL·E（示例）"
    default_url = "https://api.openai.com/v1/images/generations"
    default_model = "dall-e-3"

    def _payload(self, prompt, minimal=False):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": "256x256" if minimal else self.image_size,
            "n": 1,
            "response_format": self.response_format or "b64_json",  # 默认 b64 直返
        }
        return payload

    def generate(self, prompt, save_path, reference_image=None):
        status, result, err = self._post_json(self._payload(prompt))
        if err:
            return False, err
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or not data:
            return False, "API 返回的 data 字段为空或格式不正确"
        item = data[0] if isinstance(data[0], dict) else {}
        url, b64 = item.get("url"), item.get("b64_json")
        try:
            if url:
                self._download_and_save(url, save_path)
            elif b64:
                self._save_b64(b64, save_path)
            else:
                return False, "API 返回数据中没有图片（既无 url 也无 b64_json）"
            return True, ""
        except Exception as e:
            return False, f"下载/保存失败: {e}"

    def test_connection(self, reference_image=None):
        if not self.url:
            return False, "请先填写 API URL"
        if not self.key:
            return False, "请先填写 API Key"
        status, result, err = self._post_json(self._payload("test", minimal=True), timeout=30)
        if err:
            return False, err
        return True, f"连接成功（HTTP 200）· 模型 {self.model} 可用"


# =============================================================================
# 参考范本 2：Stability AI（text_prompts/height/width，返回 artifacts[0].base64）
# -----------------------------------------------------------------------------
# 演示：不同字段、base64 直返、非 Bearer 鉴权（api-key 头）的平台如何接入。
# 该平台真正支持图生图：reference_image 非空时改用 multipart 上传到 image-to-image 端点。
# =============================================================================
class StabilityProvider(ProviderBase):
    id = "stability"
    display_name = "Stability AI（示例）"
    default_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    default_model = "stable-diffusion-xl-1024-v1-0"

    def _size_wh(self):
        try:
            w, h = self.image_size.lower().split("x")
            return int(w), int(h)
        except Exception:
            return 1024, 1024

    def _headers(self):
        # Stability 用 api-key 头（非 Bearer），可被 extra_headers 覆盖
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        for k, v in (self.extra_headers or {}).items():
            headers[k] = str(v)
        return headers

    def _payload(self, prompt, minimal=False):
        w, h = self._size_wh()
        return {
            "text_prompts": [{"text": prompt}],
            "height": h,
            "width": w,
            "cfg_scale": self.guidance_scale,
            "steps": 1 if minimal else self.num_inference_steps,
        }

    def _image_to_image_url(self):
        """图生图端点：/v1/generation/{engine}/image-to-image（multipart）"""
        base = self.url.replace("/text-to-image", "")
        engine = base.rstrip("/").split("/")[-1]
        return f"{base}/image-to-image"

    def generate(self, prompt, save_path, reference_image=None):
        if reference_image is None:
            # 纯文生图：JSON POST
            status, result, err = self._post_json(self._payload(prompt))
        else:
            # 图生图：multipart/form-data 上传参考图（Stability 的 image-to-image 接口）
            try:
                with open(reference_image, 'rb') as f:
                    files = {"init_image": f}
                    data = {"text_prompts[0][text]": prompt}
                    resp = requests.post(self._image_to_image_url(),
                                         headers={"api-key": self.key,
                                                  **self.extra_headers},
                                         files=files, data=data, timeout=120)
                if resp.status_code != 200:
                    return False, f"HTTP {resp.status_code}: {(resp.text or '')[:200]}"
                result = resp.json()
            except Exception as e:
                return False, f"图生图请求异常: {e}"
        artifacts = result.get("artifacts") if isinstance(result, dict) else None
        if not isinstance(artifacts, list) or not artifacts:
            return False, "API 返回的 artifacts 字段为空或格式不正确"
        item = artifacts[0]
        b64 = item.get("base64") if isinstance(item, dict) else None
        if not b64:
            return False, "artifacts[0] 没有 base64 数据"
        try:
            self._save_b64(b64, save_path)
            return True, ""
        except Exception as e:
            return False, f"保存失败: {e}"

    def test_connection(self, reference_image=None):
        if not self.url:
            return False, "请先填写 API URL"
        if not self.key:
            return False, "请先填写 API Key"
        status, result, err = self._post_json(self._payload("test", minimal=True), timeout=30)
        if err:
            return False, err
        return True, f"连接成功（HTTP 200）· 模型 {self.model} 可用"


# =============================================================================
# 平台注册表 + 工厂
# =============================================================================
PROVIDER_REGISTRY = {
    "siliconflow": SiliconFlowProvider,
    "custom": CustomProvider,
    "openai": OpenAIProvider,
    "stability": StabilityProvider,
}


PLATFORM_HINTS = {
    "siliconflow": "内置平台：硅基流动 SiliconFlow。\n"
                   "只需填写 API Key 与模型名称即可使用（URL 已内置）。",
    "custom": "OpenAI 兼容文生图接口（POST /v1/images/generations）\n"
              "适用：返回 url 或 b64_json 的 OpenAI 兼容接口（通义万相、智谱 CogView、阶跃星辰等）。\n"
              "① API URL：以 /v1/images/generations 结尾（如 https://api.xxx.com/v1/images/generations）\n"
              "② API Key：平台控制台生成的密钥\n"
              "③ 模型名称：平台支持的图像模型 ID\n"
              "参考图：若你的模型支持图生图，勾选「该模型支持参考图/图生图输入」，"
              "生成时将可上传参考图（以 image 字段随请求发送）。\n"
              "高级：可在 settings.json 中配置 response_format（url/b64_json）协商返回格式，"
              "或用 extra_headers 添加自定义鉴权头（如 api-key）。",
    "openai": "示例平台：OpenAI DALL·E 官方接口（字段 size/n、默认 b64_json 返回）。\n"
              "需要 OpenAI 官方 API Key；作为「字段协议不同」的接入范本。",
    "stability": "示例平台：Stability AI（text_prompts/height/width、返回 base64、api-key 头鉴权）。\n"
                 "需要 Stability 官方 Key；作为「非 OpenAI 协议 + base64 直返」的接入范本。",
}


def get_provider_ids():
    """返回所有已注册平台的 id 列表（设置页下拉框用）"""
    return list(PROVIDER_REGISTRY.keys())


def get_provider_display_name(provider_id):
    """平台 id → 显示名（未知平台回退为 id 本身）"""
    cls = PROVIDER_REGISTRY.get(provider_id)
    return cls.display_name if cls else provider_id


def create_provider(cfg):
    """工厂：按配置中的 provider 字段创建对应适配器实例，未知平台回退到硅基流动"""
    provider_id = (cfg.get("provider") or "").strip() or "siliconflow"
    cls = PROVIDER_REGISTRY.get(provider_id, SiliconFlowProvider)
    return cls(cfg)
