"""资源模型（外部三维资产）读取与展示子系统。

与「照片重建」管线并行的一条独立能力线：直接读取美术资源库里的成品模型
（OpenFlight .flt / Performer .ive + TGA/DDS 贴图 + docx 技术说明 + 转盘 sprites），
做格式识别、元数据解析、贴图归一化，并在能可靠提取几何时导出 GLB 供 three.js 预览。

设计原则（延续本仓库零二进制入库理念）：
- 只做「读取/转换」，产物写入 assets_out/ 工作目录，原始美术资源不入库；
- 几何提取带严格校验，拿不准就干净地报告「不支持」并回退到转盘展示，绝不产出错乱网格。
"""

from __future__ import annotations

from .library import (
    AssetBundle, scan_asset_dir, import_asset_dir, import_assets_root,
    load_asset_bundle, list_library_asset_dirs, slugify, is_up_to_date,
)
from .openflight import OpenFlightInfo, describe_openflight, describe_ive, extract_mesh
from .spec import parse_model_spec
from .textures import load_image, normalize_texture
from .demo import (
    build_single_file_html, render_single_file_html,
    render_library_index, write_library_index,
)

__all__ = [
    "AssetBundle", "scan_asset_dir", "import_asset_dir", "import_assets_root",
    "load_asset_bundle", "list_library_asset_dirs", "slugify", "is_up_to_date",
    "OpenFlightInfo", "describe_openflight", "describe_ive", "extract_mesh",
    "parse_model_spec", "load_image", "normalize_texture",
    "build_single_file_html", "render_single_file_html",
    "render_library_index", "write_library_index",
]
