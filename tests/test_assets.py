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
    describe_openflight, extract_mesh,
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
    import_asset_dir(asset_dir, lib / "agm-x")
    appmod.set_data_dir(str(tmp_path / "ws"))
    appmod.set_assets_dir(str(lib))
    app = appmod.create_app()

    with TestClient(app) as c:
        items = c.get("/api/assets").json()["assets"]
        assert any(a["folder"] == "agm-x" and a["sprite_count"] == 36 for a in items)

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
    for _, o, _b in results:
        assert (o / "asset.json").is_file()
        assert (o / "sprites" / "frame_000.png").is_file()
    assert not (out / ".svn").exists()


def test_static_gallery_index(tmp_path):
    from multisight.assets import build_single_file_html
    parent = tmp_path / "src"
    _build_asset(parent / "missile-a", "导弹A")
    _build_asset(parent / "missile-b", "导弹B")
    out = tmp_path / "assets_out"
    for name, o, b in import_assets_root(parent, out):
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
