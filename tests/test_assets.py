"""资源模型读取子系统测试：合成一个「美术资源目录」跑通
scan/import/demo/GLB/server 资产接口全链路（零二进制入库，全部临时生成）。

约定目录形态（与真实鱼叉 AGM-84 资源包一致）：
    <folder>/
      model.flt                     （OpenFlight 头 + 空壳 vertexList，几何应校验失败回退）
      model_c_512.tga               （漫反射贴图）
      sprites/S.0.png … S.35.png     （36 帧转盘）
      preview.png / structure.png
      三维模型技术说明.docx           （元数据）
"""

from __future__ import annotations

import struct
import zipfile

import pytest
from PIL import Image

from multisight.assets import (
    scan_asset_dir, import_asset_dir, import_assets_root, load_asset_bundle,
    render_single_file_html, render_library_index, write_library_index,
    describe_openflight, extract_mesh, find_model_roots,
)
from multisight.assets.glb import write_glb


# ---------------------------------------------------------------- 合成资源

def _write_docx(path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>"
                   for p in paragraphs)
    doc = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                   'package/2006/content-types"/>')
        z.writestr("word/document.xml", doc)


def _write_flt(path):
    """308 字节头（大端，magic 0x0001，顶点单位=米）+ 一条全零 vertexList（空壳）。"""
    hdr = bytearray(308)
    struct.pack_into(">HH", hdr, 0, 0x0001, 308)   # magic + 头长(308→版本显示 3.8)
    hdr[4:6] = b"db"                                # ascii id
    struct.pack_into(">II", hdr, 12, 1800, 1)       # format/edit revision
    hdr[20:20 + 9] = b"2013-01-02"                  # birth
    hdr[62] = 0                                     # 顶点单位 0=米
    # vertexList：opcode=5，count=3，顶点全零（触发空壳判定 → extract 返回 None）
    rec = bytearray(44)
    struct.pack_into(">HH", rec, 0, 5, 44)
    struct.pack_into(">H", rec, 4, 3)               # count
    # 其余 12*3 字节保持 0（顶点坐标全零）
    path.write_bytes(bytes(hdr) + bytes(rec))


def _build_asset(d, display_name):
    """在目录 d 里合成一个完整资源模型（flt + tga + 36 帧 sprites + 渲染图 + docx）。"""
    (d / "sprites").mkdir(parents=True, exist_ok=True)
    _write_flt(d / "model.flt")
    Image.new("RGB", (48, 48), (200, 40, 40)).save(d / "model_c_512.tga")
    for i in range(36):
        Image.new("RGBA", (16, 16), (i * 5 % 256, 60, 120, 255)).save(
            d / "sprites" / f"S.{i}.png")
    Image.new("RGB", (64, 48), (10, 200, 30)).save(d / "preview.png")
    Image.new("RGB", (64, 48), (30, 10, 200)).save(d / "structure.png")
    _write_docx(d / "三维模型技术说明.docx", [
        "模型名称", display_name,
        "模型面数", "1092",
        "模型文件格式", "Flt/ive",
        "比例尺", "米",
        "坐标系与原点说明",
        "坐标系Y朝向实体正前方，Z朝向实体正上方，X轴向符合右手定则。",
        "贴图：model_c_512 大小：512*512像素（文件 model_c_512.tga）",
    ])
    return d


@pytest.fixture()
def asset_dir(tmp_path):
    return _build_asset(tmp_path / "agm-x", "测试反舰导弹-agm-x")


# ---------------------------------------------------------------- 扫描/导入

def test_scan_asset_dir(asset_dir):
    b = scan_asset_dir(asset_dir)
    assert b.name == "测试反舰导弹-agm-x"
    assert b.meta["triangles"] == 1092
    assert b.meta["format"] == "Flt/ive"
    assert b.meta["scale"] == "米"
    assert "右手" in b.meta["coordinate_system"]
    assert b.meta["texture_file"] == "model_c_512.tga"
    # OpenFlight 头可靠解析
    assert b.openflight["ok"] is True
    assert b.openflight["version_str"] == "3.8"
    assert b.openflight["vertex_unit"] == "米"
    # 空壳 vertexList → 几何校验诚实失败，回退转盘
    assert b.geometry_ok is False
    assert b.triangles == 0
    assert len(b.sprites) == 36
    assert b.texture_png and b.texture_png.endswith("model_c_512.tga")
    assert b.display == "turntable"


def test_import_and_roundtrip(asset_dir, tmp_path):
    out = tmp_path / "library" / "agm-x"
    b = import_asset_dir(asset_dir, out)
    assert (out / "asset.json").is_file()
    assert (out / "texture.png").is_file()          # TGA 归一化为 PNG
    assert (out / "sprites" / "frame_000.png").is_file()
    assert b.sprites[0].endswith("frame_000.png")
    assert b.glb is None                             # 几何未通过 → 无 GLB
    # asset.json 存相对路径，加载后解析回绝对且文件都在
    rb = load_asset_bundle(out)
    assert rb.name == b.name
    from pathlib import Path
    assert Path(rb.texture_png).is_file()
    assert all(Path(s).is_file() for s in rb.sprites)


def test_single_file_demo_self_contained(asset_dir, tmp_path):
    out = tmp_path / "lib" / "agm-x"
    b = import_asset_dir(asset_dir, out)
    html = render_single_file_html(load_asset_bundle(out))
    assert "base64," in html
    assert "测试反舰导弹-agm-x" in html
    # 离线自包含：不应出现 http(s) 资源引用（转盘/贴图路径全 data URI）
    assert 'src="http' not in html and "url(http" not in html


def test_demo_turntable_scrubber_keyboard(asset_dir, tmp_path):
    out = tmp_path / "lib" / "agm-x"
    import_asset_dir(asset_dir, out)
    html = render_single_file_html(load_asset_bundle(out))
    assert 'id="scrub"' in html                    # 帧滑杆
    assert "ArrowRight" in html and "Home" in html and "End" in html   # 键盘逐帧


# ---------------------------------------------------------------- 底层单元

def test_describe_bad_magic():
    info = describe_openflight(b"\x00\x02" + b"\x00" * 400)
    assert info.ok is False


def test_extract_mesh_empty_shell_returns_none():
    # 308 头 + 全零 vertexList（同 fixture）应拒绝
    buf = bytearray(308)
    struct.pack_into(">HH", buf, 0, 0x0001, 308)
    rec = bytearray(44)
    struct.pack_into(">HH", rec, 0, 5, 44)
    struct.pack_into(">H", rec, 4, 3)
    assert extract_mesh(bytes(buf) + bytes(rec)) is None


def test_write_glb_magic(tmp_path):
    positions = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    indices = [0, 1, 2, 0, 2, 3]
    p = write_glb(tmp_path / "m.glb", positions, indices, name="quad")
    data = p.read_bytes()
    assert data[:4] == b"glTF"
    # JSON chunk + BIN chunk 总长按 4 字节对齐
    json_len = struct.unpack_from("<I", data, 12)[0]
    assert json_len % 4 == 0


# ---------------------------------------------------------------- server 接口

def test_server_asset_endpoints(asset_dir, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from multisight.server import app as appmod

    lib = tmp_path / "library"
    import_asset_dir(asset_dir, lib / "agm-x", category="反舰导弹")
    appmod.set_data_dir(str(tmp_path / "ws"))
    appmod.set_assets_dir(str(lib))
    app = appmod.create_app()

    with TestClient(app) as c:
        items = c.get("/api/assets").json()["assets"]
        hit = next(a for a in items if a["folder"] == "agm-x")
        assert hit["sprite_count"] == 36
        assert hit["category"] == "反舰导弹"           # 列表 API 暴露分类（供 WebUI 分组）

        r = c.get("/api/assets/agm-x/demo")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "base64," in r.text

        f = c.get("/api/assets/agm-x/file", params={"path": "texture.png"})
        assert f.status_code == 200 and f.content[:8] == b"\x89PNG\r\n\x1a\n"

        # 路径穿越守卫 + 未知资产
        assert c.get("/api/assets/agm-x/file", params={"path": "../../etc/passwd"}).status_code == 404
        assert c.get("/api/assets/nope/demo").status_code == 404


# ---------------------------------------------------------------- 批量导入 / 画廊

def test_import_assets_root_skips_non_asset(tmp_path):
    parent = tmp_path / "library_src"
    _build_asset(parent / "missile-a", "导弹A")
    _build_asset(parent / "missile-b", "导弹B")
    (parent / ".svn" / "pristine").mkdir(parents=True)   # 非资源目录应被跳过
    out = tmp_path / "assets_out"
    results = import_assets_root(parent, out)
    assert {r[0] for r in results} == {"missile-a", "missile-b"}
    for _, o, _b, _sk in results:
        assert (o / "asset.json").is_file()
        assert (o / "sprites" / "frame_000.png").is_file()
    assert not (out / ".svn").exists()


def test_static_gallery_index(tmp_path):
    from multisight.assets import build_single_file_html
    parent = tmp_path / "src"
    _build_asset(parent / "missile-a", "导弹A")
    _build_asset(parent / "missile-b", "导弹B")
    out = tmp_path / "assets_out"
    for name, o, b, _sk in import_assets_root(parent, out):
        build_single_file_html(b, o / "demo.html")
    html = write_library_index(out).read_text(encoding="utf-8")
    assert "导弹A" in html and "导弹B" in html
    assert "missile-a/demo.html" in html and "missile-b/demo.html" in html
    assert 'src="missile-a/preview.png"' in html
    # 静态画廊用相对路径，不含网络/绝对地址 → 可整目录托管或双击打开
    assert "http" not in html


def test_render_library_index_api(tmp_path):
    out = tmp_path / "assets_out"
    for n in ("aa", "bb"):
        _build_asset(tmp_path / "src" / n, "名称-" + n)
        import_asset_dir(tmp_path / "src" / n, out / n)
    html = render_library_index(out, api=True)
    assert "/api/assets/aa/demo" in html
    assert "/api/assets/bb/file?path=" in html


def test_server_gallery(asset_dir, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from multisight.server import app as appmod

    lib = tmp_path / "library"
    import_asset_dir(asset_dir, lib / "agm-x")
    appmod.set_data_dir(str(tmp_path / "ws"))
    appmod.set_assets_dir(str(lib))
    with TestClient(appmod.create_app()) as c:
        r = c.get("/api/assets/gallery")
        assert r.status_code == 200
        assert "/api/assets/agm-x/demo" in r.text
        assert "/api/assets/agm-x/file?path=" in r.text


# ---------------------------------------------------------------- 画廊交互 / 重导入稳健性

def test_gallery_has_search_filter_ui(tmp_path):
    out = tmp_path / "assets_out"
    _build_asset(tmp_path / "src" / "a", "红导弹")
    _build_asset(tmp_path / "src" / "b", "蓝鱼雷")
    import_asset_dir(tmp_path / "src" / "a", out / "a")
    import_asset_dir(tmp_path / "src" / "b", out / "b")
    html = render_library_index(out, api=False)
    assert 'id="q"' in html                     # 搜索框
    assert "全部 <b>2</b>" in html               # 总数 chip
    assert 'data-t="turntable"' in html          # 类型筛选属性
    assert "显示" in html                        # 计数 JS
    assert 'id="sort"' in html                   # 排序下拉
    assert 'data-frames="36"' in html            # 排序数据属性
    assert 'data-tris=' in html
    assert "syncHash" in html and "location.hash" in html  # 可分享状态（URL hash）
    assert 'id="nores"' in html                    # 无匹配空态
    assert "hashchange" in html                    # 前进/后退也能恢复
    assert "http" not in html                    # 仍零网络自包含


def test_reimport_clears_stale_frames(asset_dir, tmp_path):
    from pathlib import Path
    out = tmp_path / "lib" / "agm-x"
    import_asset_dir(asset_dir, out)
    stale = out / "sprites" / "frame_999.png"
    stale.write_bytes(b"junk")                  # 模拟上次导入多出来的陈旧帧
    b2 = import_asset_dir(asset_dir, out)
    assert not stale.exists()                    # 重导入应清空重建 sprites
    assert len(b2.sprites) == 36
    assert all(Path(s).name.startswith("frame_0") for s in b2.sprites)


# ---------------------------------------------------------------- Performer .ive 指纹

def test_describe_ive_signature():
    from multisight.assets import describe_ive
    good = bytes([4, 3, 2, 1]) + b"\x00\x00\x124" + b"\x00" * 8
    d = describe_ive(good)
    assert d["ok"] is True and d["size"] == len(good)
    assert describe_ive(b"NOPE" + b"\x00" * 8)["ok"] is False


def test_scan_detects_ive(asset_dir):
    # 向资源目录追加一个 .ive，scan 应识别 Performer 指纹（不做几何）
    (asset_dir / "agm.ive").write_bytes(bytes([4, 3, 2, 1]) + b"\x00\x00\x00\x01" + b"\x00" * 8)
    b = scan_asset_dir(asset_dir)
    assert b.performer.get("ok") is True
    assert b.performer["size"] == 16


# ---------------------------------------------------------------- 完整性 warnings

def test_scan_warnings(asset_dir):
    b = scan_asset_dir(asset_dir)
    assert any("几何" in w for w in b.warnings)          # flt 几何未通过 → 提醒
    assert not any("缺技术说明" in w for w in b.warnings)  # 完整目录不应报缺文档
    assert not any("未找到" in w for w in b.warnings)


def test_warnings_shown_in_gallery(tmp_path):
    out = tmp_path / "assets_out"
    d = _build_asset(tmp_path / "src" / "only", "裸模型")
    (d / "三维模型技术说明.docx").unlink()             # 制造缺文档/缺贴图
    (d / "model_c_512.tga").unlink()
    b = import_asset_dir(d, out / "only")
    assert any("缺技术说明" in w for w in b.warnings)
    assert any("未找到" in w for w in b.warnings)
    assert "⚠" in render_library_index(out)            # 画廊卡片带警示徒章


# ---------------------------------------------------------------- DDS 贴图支持

def _save_dds(path, mode="RGBA", color=(200, 30, 30, 255)):
    im = Image.new(mode, (16, 16), color)
    try:
        im.save(path, "DDS")
    except (ValueError, OSError):
        pytest.skip("此 Pillow 构建不支持 DDS")
    return path


def test_normalize_dds_to_png(tmp_path):
    from multisight.assets.textures import normalize_texture
    dds = _save_dds(tmp_path / "t.dds")
    out = normalize_texture(dds, tmp_path / "out" / "texture.png")
    assert out is not None and out.exists()
    assert Image.open(out).size == (16, 16)


def test_scan_picks_dds_texture(tmp_path):
    # 一个只有 .flt + .dds 贴图（无 tga）的目录，scan 应把 .dds 认作漫反射贴图
    d = tmp_path / "ddsasset"
    (d).mkdir()
    _write_flt(d / "m.flt")
    _save_dds(d / "diffuse.dds")
    b = scan_asset_dir(d)
    assert b.texture_png and b.texture_png.lower().endswith(".dds")
    b2 = import_asset_dir(d, tmp_path / "lib" / "ddsasset")
    assert b2.texture_png.endswith("texture.png")       # 已归一化为 PNG
    assert (tmp_path / "lib" / "ddsasset" / "texture.png").is_file()


# ---------------------------------------------------------------- 批量增量导入

def test_batch_incremental_skips_unchanged(tmp_path):
    parent = tmp_path / "library_src"
    _build_asset(parent / "m1", "导弹1")
    out = tmp_path / "assets_out"
    r1 = import_assets_root(parent, out)
    assert all(not sk for _, _, _, sk in r1)            # 首次全部实际导入
    marker = out / "m1" / "sprites" / "MARKER.txt"
    marker.write_text("keep")                          # 重导会先 rmtree sprites 而擦掉它
    r2 = import_assets_root(parent, out)
    assert all(sk for _, _, _, sk in r2)                # 未变化 → 全部跳过
    assert marker.is_file()                             # 跳过 → 产物不被重建
    r3 = import_assets_root(parent, out, force=True)
    assert all(not sk for _, _, _, sk in r3)            # --force → 强制重导
    assert not marker.exists()                          # 重导清空了陈旧帧


def test_batch_reimport_on_source_change(tmp_path):
    from multisight.assets import is_up_to_date
    parent = tmp_path / "library_src"
    d = _build_asset(parent / "m1", "导弹1")
    out = tmp_path / "assets_out"
    import_assets_root(parent, out)
    assert is_up_to_date(out / "m1", d)                 # 刚导完应判定为最新
    (d / "model_c_512.tga").write_bytes(b"x" * 5000)   # 改贴图大小 → 签名变化
    assert not is_up_to_date(out / "m1", d)
    r = import_assets_root(parent, out)
    assert r[0][3] is False                             # 变化→ 重导（未跳过）


def test_batch_error_isolation(tmp_path, monkeypatch):
    """整库无人值守批量：单个型号导入抛错不应中断整批，坏条目以警告占位。"""
    from pathlib import Path
    import multisight.assets.library as lib
    parent = tmp_path / "library_src"
    _build_asset(parent / "m1", "导弹1")
    _build_asset(parent / "boom", "坏型号")
    _build_asset(parent / "m2", "导弹2")
    real = lib.import_asset_dir

    def flaky(folder, out_dir, *, category=""):
        if Path(folder).name == "boom":
            raise RuntimeError("模拟磁盘/解码故障")
        return real(folder, out_dir, category=category)

    monkeypatch.setattr(lib, "import_asset_dir", flaky)
    out = tmp_path / "assets_out"
    results = lib.import_assets_root(parent, out)
    names = {r[0] for r in results}
    assert names == {"m1", "boom", "m2"}               # 三个都返回，未中断
    bad = next(r for r in results if r[0] == "boom")
    assert any("导入失败" in w for w in bad[2].warnings)
    assert not (out / "boom" / "asset.json").is_file()  # 坏条目无产物 → 画廊自动忽略
    for name in ("m1", "m2"):                           # 其余型号正常导入
        good = next(r for r in results if r[0] == name)
        assert (good[1] / "asset.json").is_file()
        assert not any("导入失败" in w for w in good[2].warnings)


# ---------------------------------------------------------------- 分类发现 / 整库导入

def test_find_model_roots_whole_library(tmp_path):
    lib = tmp_path / "三维模型"
    _build_asset(lib / "战斗机" / "F16战隼", "F16")
    _build_asset(lib / "战斗机" / "F22猛禽", "F22")
    _build_asset(lib / "潜艇" / "元27", "元27")
    (lib / "战斗机" / ".svn").mkdir(parents=True)      # 隐藏目录应被跳过
    roots = find_model_roots(lib)
    got = {(leaf.name, cat, prefix) for leaf, cat, prefix in roots}
    assert got == {("F16战隼", "战斗机", True), ("F22猛禽", "战斗机", True),
                   ("元27", "潜艇", True)}


def test_find_model_roots_per_category(tmp_path):
    cat_dir = tmp_path / "反舰导弹"
    d = _build_asset(cat_dir / "鱼叉AGM84", "鱼叉")
    roots = find_model_roots(cat_dir)
    assert roots == [(d, "反舰导弹", False)]             # 直接子型号：大类=父名、不前缀


def test_whole_library_import_category(tmp_path):
    lib = tmp_path / "三维模型"
    _build_asset(lib / "战斗机" / "F16", "F16")
    _build_asset(lib / "潜艇" / "元27", "元27")
    out = tmp_path / "assets_out"
    import_assets_root(lib, out)
    # 目录名带大类前缀，防重名；asset.json 记录 category
    assert (out / "战斗机-F16" / "asset.json").is_file()
    assert (out / "潜艇-元27" / "asset.json").is_file()
    assert load_asset_bundle(out / "战斗机-F16").category == "战斗机"
    html = render_library_index(out, api=False)
    assert 'id="catf"' in html                         # 多分类→分类下拉
    assert 'data-cat="战斗机"' in html
    assert "http" not in html                          # 仍零网络


def test_single_category_no_filter(tmp_path):
    cat_dir = tmp_path / "反舰导弹"
    _build_asset(cat_dir / "鱼叉", "鱼叉")
    out = tmp_path / "assets_out"
    import_assets_root(cat_dir, out)
    html = render_library_index(out, api=False)
    assert 'id="catf"' not in html                     # 单一分类不给下拉


# ---------------------------------------------------------------- docx 字段提取加固

def test_spec_coordinate_on_separate_line(tmp_path):
    """真实常见：描述行不带“坐标”二字（只说「…符合右手定则」），标题单独一段。"""
    from multisight.assets import parse_model_spec
    d = _write_docx(tmp_path / "s.docx", [
        "坐标系与原点说明",                              # 标题（不应被当作值）
        "Y朝向实体正前方，Z朝向实体正上方，X轴向符合右手定则。",
        "坐标原点在世界坐标中心",
    ])
    meta = parse_model_spec(tmp_path / "s.docx")
    assert "右手定则" in meta["coordinate_system"]
    assert "实体正前方" in meta["coordinate_system"]
    assert "世界坐标中心" in meta["origin"]
    # 标题行不会被误当成坐标值
    assert meta["coordinate_system"] != "坐标系与原点说明"


def test_demo_surfaces_origin(asset_dir):
    """带原点描述的 docx → demo 元数据面板应多出「原点」行。"""
    _write_docx(asset_dir / "三维模型技术说明.docx", [
        "模型名称", "带原点型号",
        "模型面数", "1200",
        "坐标系与原点说明",
        "Y朝向实体正前方，符合右手定则。",
        "坐标原点在世界坐标中心",
    ])
    from multisight.assets import scan_asset_dir as _scan
    b = _scan(asset_dir)
    assert b.meta["origin"] == "坐标原点在世界坐标中心"
    html = render_single_file_html(b)
    assert "原点" in html and "世界坐标中心" in html


def test_demo_surfaces_category(asset_dir):
    """整库导入带分类 → 单文件 demo 元数据面板应自述所属大类。"""
    from multisight.assets import import_asset_dir as _imp
    out = asset_dir.parent / "lib-out"
    b = _imp(asset_dir, out, category="战斗机")
    html = render_single_file_html(b)
    assert "分类" in html and "战斗机" in html


def test_spec_texture_name_without_extension(tmp_path):
    """常见变体：docx 只写“贴图：feiji_c_01”（无扩展名）→ 也应抽出 texture_file。"""
    from multisight.assets import parse_model_spec
    _write_docx(tmp_path / "s.docx", [
        "模型贴图说明",
        "贴图：feiji_c_01  大小：1024*1024像素",
    ])
    meta = parse_model_spec(tmp_path / "s.docx")
    assert meta["texture_file"] == "feiji_c_01"
    assert "1024*1024" in meta["texture_note"]


def test_scan_matches_texture_by_stem(tmp_path):
    """声明名为基名（无扩展）时，按 stem 优先匹配声明贴图，而非盲选最大图。"""
    from pathlib import Path
    from multisight.assets import scan_asset_dir as _scan
    d = tmp_path / "m"
    d.mkdir(parents=True)
    _write_flt(d / "model.flt")
    # declared 贴图很小；decoy 很大会被“最大图”回退误选
    Image.new("RGB", (8, 8), (200, 30, 30)).save(d / "feiji_c_01.tga")
    Image.new("RGB", (64, 64), (10, 10, 10)).save(d / "other_big.tga")
    _write_docx(d / "三维模型技术说明.docx", [
        "模型名称", "按声明匹配",
        "贴图：feiji_c_01 大小：8*8像素",
    ])
    b = _scan(d)
    assert b.meta["texture_file"] == "feiji_c_01"
    assert Path(b.texture_png).name == "feiji_c_01.tga"   # 命中声明而非更大 decoy
