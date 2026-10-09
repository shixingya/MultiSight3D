"""单文件自包含 HTML demo 生成器。

把 AssetBundle 的转盘帧 / 贴图 / 渲染图全部以 base64 data URI 内联进一个 .html，
零依赖、零网络即可「双击打开直接运行」，正合「无需部署，线上直接运行」诉求。

三种展示模式（按 bundle.display 自动选择，并全部作为可切换标签呈现）：
- turntable：36 帧转盘（拖拽旋转 + 自动旋转 + 滚轮缩放），真实模型渲染，离线可用；
- glb：若 .flt 几何校验通过，内嵌 GLB + three.js（CDN，离线时给出降级提示）真三维轨道查看；
- texture：漫反射 UV 贴图查看。
外加「元数据」面板（docx 技术说明 + OpenFlight 格式指纹）。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .library import AssetBundle, load_asset_bundle, list_library_asset_dirs


def _b64(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    mime = "image/png" if p.suffix.lower() == ".png" else (
        "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "application/octet-stream")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def build_single_file_html(bundle: AssetBundle, out_path: Path | str,
                           *, glb_path: str | None = None) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_single_file_html(bundle, glb_path=glb_path), encoding="utf-8")
    return out


def _thumb_rel(folder: Path) -> str | None:
    """为画廊卡片选一张相对 folder 的缩略图（preview > structure > 首帧 > texture）。"""
    for cand in ("preview.png", "structure.png"):
        if (folder / cand).is_file():
            return cand
    sp = folder / "sprites"
    if sp.is_dir():
        frames = sorted(sp.iterdir())
        if frames:
            return f"sprites/{frames[0].name}"
    if (folder / "texture.png").is_file():
        return "texture.png"
    return None


def render_library_index(root: Path | str, *, api: bool = False) -> str:
    """渲染资源模型画廊 index.html（卡片网格 + 缩略图 + 链接到每个资产的 demo）。

    - api=False（静态托管 / 本地双击）：链接 <folder>/demo.html，缩略图 <folder>/<rel>（相对路径）；
    - api=True（Web 服务内）：链接 /api/assets/<folder>/demo，缩略图 /api/assets/<folder>/file?path=<rel>。
    """
    root = Path(root)
    cards = []
    counts: dict[str, int] = {}
    for folder in list_library_asset_dirs(root):
        try:
            b = load_asset_bundle(folder)
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        disp = {"glb": "真三维", "turntable": "转盘", "texture": "贴图"}.get(b.display, b.display)
        counts[b.display] = counts.get(b.display, 0) + 1
        pills = [disp, f"{len(b.sprites)} 帧"]
        if b.triangles:
            pills.append(f"{b.triangles} 面")
        rel = _thumb_rel(folder)
        if api:
            link = f"/api/assets/{quote(folder.name)}/demo"
            thumb = f"/api/assets/{quote(folder.name)}/file?path={quote(rel)}" if rel else ""
        else:
            link = f"{quote(folder.name)}/demo.html"
            thumb = f"{quote(folder.name)}/{quote(rel)}" if rel else ""
        pic = (f'<img loading="lazy" src="{thumb}" alt=""/>' if thumb
               else '<div class="noimg">无预览图</div>')
        raw_title = b.name or folder.name
        title = raw_title.replace("&", "&amp;").replace("<", "&lt;")
        searchable = raw_title.lower().replace('"', "&quot;")
        cards.append(
            f'<a class="card" href="{link}" data-t="{b.display}" data-name="{searchable}">'
            f'{pic}<div class="meta">'
            f"<h3>{title}</h3><div class=\"pills\">"
            + "".join(f"<span>{p}</span>" for p in pills) + "</div></div></a>")
    grid = "\n".join(cards) or '<p class="empty">暂无资源模型，先用 import-asset 导入。</p>'
    total = sum(counts.values())
    chips = ['<button class="chip active" data-f="">全部 <b>' + str(total) + "</b></button>"]
    for key, label in (("glb", "真三维"), ("turntable", "转盘"), ("texture", "贴图")):
        if counts.get(key):
            chips.append(f'<button class="chip" data-f="{key}">{label} <b>{counts[key]}</b></button>')
    html = _INDEX_TEMPLATE.replace("__GRID__", grid)
    html = html.replace("__CHIPS__", "".join(chips))
    return html


def write_library_index(root: Path | str, out_path: Path | str | None = None) -> Path:
    """把静态画廊写到 <root>/index.html（可整个目录丢到任意静态托管 / GitHub Pages）。"""
    root = Path(root)
    out = Path(out_path) if out_path else root / "index.html"
    out.write_text(render_library_index(root, api=False), encoding="utf-8")
    return out


def render_single_file_html(bundle: AssetBundle, *, glb_path: str | None = None) -> str:
    frames = [b for b in (_b64(s) for s in bundle.sprites) if b]
    texture = _b64(bundle.texture_png)
    preview = _b64(bundle.preview)
    structure = _b64(bundle.structure)
    glb_uri = _b64(glb_path or bundle.glb)

    meta = bundle.meta or {}
    of = bundle.openflight or {}

    def esc(v: Any) -> str:
        return ("" if v is None else str(v)).replace("&", "&amp;").replace("<", "&lt;")

    rows = []
    for label, key in (("模型名称", "name"), ("三角面数", "triangles"), ("源格式", "format"),
                       ("比例尺", "scale"), ("坐标系", "coordinate_system"), ("贴图", "texture_note")):
        if meta.get(key):
            rows.append(f"<tr><th>{label}</th><td>{esc(meta[key])}</td></tr>")
    for label, key in (("OpenFlight 版本", "version_str"), ("顶点单位", "vertex_unit"),
                       ("生成时间", "birth"), ("记录数", "record_count")):
        if of.get(key):
            rows.append(f"<tr><th>{label}(flt)</th><td>{esc(of[key])}</td></tr>")
    geom = ("✅ 几何已可靠提取，可切「真三维」" if bundle.geometry_ok
            else "⚠️ 该 .flt 为非标准导出变体，几何未通过校验 → 采用真实渲染转盘展示")
    rows.append(f"<tr><th>几何状态</th><td>{esc(geom)}</td></tr>")
    meta_table = "\n".join(rows)

    tabs = [("turntable", "🔄 360° 转盘")] if frames else []
    if glb_uri:
        tabs.append(("glb", "🧊 真三维"))
    if texture:
        tabs.append(("texture", "🎨 贴图"))
    if preview:
        tabs.append(("preview", "🖼 渲染图"))
    tabs.append(("meta", "ℹ 元数据"))
    tab_html = "\n".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-t="{k}">{lbl}</button>'
        for i, (k, lbl) in enumerate(tabs))

    default_tab = tabs[0][0]
    frames_json = "[" + ",".join(f'"{u}"' for u in frames) + "]"

    html = _TEMPLATE
    html = html.replace("__TITLE__", esc(bundle.name))
    html = html.replace("__TABS__", tab_html)
    html = html.replace("__DEFAULT_TAB__", default_tab)
    html = html.replace("__FRAMES__", frames_json)
    html = html.replace("__TEXTURE__", texture or "")
    html = html.replace("__PREVIEW__", preview or "")
    html = html.replace("__STRUCTURE__", structure or "")
    html = html.replace("__GLB__", glb_uri or "")
    html = html.replace("__META_TABLE__", meta_table)
    return html


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TITLE__ · MultiSight3D 资源模型</title>
<style>
 :root{--bg:#0b0d12;--panel:#141824;--line:#232a3d;--text:#dde4f2;--dim:#8b95ad;--acc:#5b8def}
 *{box-sizing:border-box;margin:0}
 body{background:var(--bg);color:var(--text);font:14px/1.6 "Segoe UI",system-ui,sans-serif;min-height:100vh}
 header{padding:16px 24px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
 header h1{font-size:18px} header span{color:var(--dim);font-size:12px}
 .tabs{display:flex;gap:8px;padding:14px 24px 0;flex-wrap:wrap}
 .tab{background:#1b2233;color:var(--dim);border:1px solid var(--line);border-radius:9px 9px 0 0;padding:8px 16px;cursor:pointer;font:inherit}
 .tab.active{color:var(--text);background:var(--panel);border-bottom-color:var(--panel)}
 main{padding:0 24px 30px}
 .pane{display:none;background:var(--panel);border:1px solid var(--line);border-radius:0 12px 12px 12px;padding:18px;min-height:420px}
 .pane.active{display:block}
 #stage{position:relative;height:440px;display:grid;place-items:center;overflow:hidden;cursor:grab;
   background:radial-gradient(circle at 50% 40%,#182034,#0a0c11 70%);border-radius:10px;user-select:none;touch-action:none}
 #stage.grab{cursor:grabbing}
 #tt{max-width:92%;max-height:92%;image-rendering:auto;transition:transform .05s;pointer-events:none}
 .hint{color:var(--dim);font-size:12px;margin-top:10px;text-align:center}
 img.full{max-width:100%;max-height:520px;display:block;margin:auto;background:#fff;border-radius:8px}
 .checker{background:repeating-conic-gradient(#20283c 0% 25%,#161c2b 0% 50%) 50%/24px 24px}
 #glbbox{height:440px;border-radius:10px;background:#08090d;position:relative;overflow:hidden}
 #glbhint{position:absolute;inset:0;display:grid;place-items:center;color:var(--dim);text-align:center;padding:20px}
 table{border-collapse:collapse;width:100%;max-width:680px}
 th,td{border:1px solid var(--line);padding:9px 12px;text-align:left;vertical-align:top;font-size:13px}
 th{color:var(--dim);width:150px;font-weight:600;background:#10151f}
</style></head>
<body>
<header><h1>__TITLE__</h1><span>MultiSight3D · 外部资源模型读取与展示 · 单文件离线可运行</span></header>
<div class="tabs">__TABS__</div>
<main>
 <div class="pane" id="p-turntable"><div id="stage"><img id="tt" alt=""/></div>
   <div class="hint">按住左右拖拽旋转 · 滚轮缩放 · 双击切换自动旋转（36 帧真实渲染）</div></div>
 <div class="pane" id="p-glb"><div id="glbbox"><div id="glbhint">加载 three.js 与内嵌 GLB…（首次需联网）</div></div></div>
 <div class="pane" id="p-texture"><img class="full checker" id="teximg" alt="texture"/></div>
 <div class="pane" id="p-preview"><img class="full" id="pvimg" alt="preview"/></div>
 <div class="pane" id="p-meta"><table>__META_TABLE__</table></div>
</main>
<script>
const FRAMES=__FRAMES__, TEXTURE="__TEXTURE__", PREVIEW="__PREVIEW__", STRUCTURE="__STRUCTURE__", GLB="__GLB__";
const DEF="__DEFAULT_TAB__";
// ---- tabs ----
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); const p=document.getElementById('p-'+b.dataset.t); p.classList.add('active');
  if(b.dataset.t==='glb'&&!window.__glbInit) initGLB();
});
document.getElementById('p-'+DEF).classList.add('active');
document.querySelector('.tab[data-t="'+DEF+'"]').classList.add('active');
if(TEXTURE) document.getElementById('teximg').src=TEXTURE;
const pv=document.getElementById('pvimg'); pv.src=PREVIEW||STRUCTURE||TEXTURE;
// ---- turntable ----
(function(){
  if(!FRAMES.length){document.getElementById('stage').innerHTML='<div style="color:#8b95ad">无转盘帧</div>';return;}
  const imgs=FRAMES.map(u=>{const i=new Image();i.src=u;return i;});
  const tt=document.getElementById('tt'), stage=document.getElementById('stage');
  let f=0, scale=1, auto=true, dragging=false, lastX=0, acc=0;
  function show(){ tt.src=FRAMES[((f%FRAMES.length)+FRAMES.length)%FRAMES.length]; }
  show();
  stage.onpointerdown=e=>{dragging=true;auto=false;lastX=e.clientX;stage.classList.add('grab');stage.setPointerCapture(e.pointerId);};
  stage.onpointermove=e=>{ if(!dragging)return; const dx=e.clientX-lastX; lastX=e.clientX; acc+=dx;
    f=Math.round(acc/10); show(); };
  stage.onpointerup=()=>{dragging=false;stage.classList.remove('grab');};
  stage.ondblclick=()=>{auto=!auto;};
  stage.onwheel=e=>{e.preventDefault();scale=Math.min(3,Math.max(.4,scale*(e.deltaY<0?1.1:0.9)));tt.style.transform='scale('+scale+')';};
  setInterval(()=>{ if(auto&&!dragging){acc+=10;f=Math.round(acc/10);show();} },70);
})();
// ---- real 3D (three.js from CDN, graceful offline) ----
async function initGLB(){
  window.__glbInit=true; const hint=document.getElementById('glbhint'), box=document.getElementById('glbbox');
  if(!GLB){hint.textContent='无可用 GLB';return;}
  let THREE,GLTFLoader,OrbitControls;
  try{
    THREE=await import('https://unpkg.com/three@0.160.0/build/three.module.js');
    ({GLTFLoader}=await import('https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js'));
    ({OrbitControls}=await import('https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js'));
  }catch(e){ hint.innerHTML='three.js 需联网加载（unpkg CDN）<br/>离线环境请改用「360° 转盘」标签'; return; }
  hint.style.display='none';
  const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(2,devicePixelRatio));
  renderer.setSize(box.clientWidth,box.clientHeight); box.appendChild(renderer.domElement);
  const scene=new THREE.Scene(); scene.background=new THREE.Color(0x08090d);
  const cam=new THREE.PerspectiveCamera(45,box.clientWidth/box.clientHeight,.01,1000); cam.position.set(3,2,4);
  scene.add(new THREE.AmbientLight(0xffffff,.8)); const d=new THREE.DirectionalLight(0xffffff,1.2); d.position.set(4,6,3); scene.add(d);
  const grid=new THREE.GridHelper(10,20,0x2a3550,0x1a2236); scene.add(grid);
  const ctl=new OrbitControls(cam,renderer.domElement); ctl.enableDamping=true; ctl.autoRotate=true;
  const loader=new GLTFLoader();
  loader.parse(await (await fetch(GLB)).arrayBuffer(), '', g=>{
    g.scene.traverse(o=>{ if(o.isMesh&&!o.geometry.attributes.normal)o.geometry.computeVertexNormals(); });
    scene.add(g.scene);
    const b=new THREE.Box3().setFromObject(g.scene), s=new THREE.Vector3(); b.getSize(s);
    const c=b.getCenter(new THREE.Vector3()); ctl.target.copy(c);
    const r=Math.max(s.x,s.y,s.z); cam.position.set(c.x+r*1.6,c.y+r*0.9,c.z+r*2.0);
  }, err=>{ hint.style.display='grid'; hint.textContent='GLB 解析失败：'+err; });
  (function loop(){requestAnimationFrame(loop);ctl.update();renderer.render(scene,cam);})();
  addEventListener('resize',()=>{renderer.setSize(box.clientWidth,box.clientHeight);cam.aspect=box.clientWidth/box.clientHeight;cam.updateProjectionMatrix();});
}
</script>
</body></html>"""


_INDEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MultiSight3D · 资源模型画廊</title>
<style>
 :root{--bg:#0b0d12;--panel:#141824;--line:#232a3d;--text:#dde4f2;--dim:#8b95ad;--acc:#5b8def}
 *{box-sizing:border-box;margin:0}
 body{background:var(--bg);color:var(--text);font:14px/1.6 "Segoe UI",system-ui,sans-serif;min-height:100vh}
 header{padding:20px 28px;border-bottom:1px solid var(--line)}
 header h1{font-size:20px} header span{color:var(--dim);font-size:12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:18px;padding:24px 28px}
 a.card{display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--line);
   border-radius:14px;overflow:hidden;text-decoration:none;color:inherit;transition:transform .12s,border-color .12s}
 a.card:hover{transform:translateY(-3px);border-color:var(--acc)}
 a.card img{width:100%;height:150px;object-fit:cover;background:#0a0c11;display:block}
 .noimg{width:100%;height:150px;display:grid;place-items:center;color:var(--dim);background:#10151f}
 .meta{padding:12px 14px}
 .meta h3{font-size:14px;font-weight:600;margin-bottom:8px;word-break:break-all}
 .pills{display:flex;flex-wrap:wrap;gap:6px}
 .pills span{background:#1b2233;color:var(--dim);border:1px solid var(--line);border-radius:6px;
   padding:2px 8px;font-size:11px}
 .empty{color:var(--dim);padding:40px 28px}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:14px 28px;border-bottom:1px solid var(--line)}
 .bar input{flex:1 1 200px;min-width:160px;background:#0f131d;color:var(--text);border:1px solid var(--line);
   border-radius:9px;padding:8px 12px;font:inherit}
 .chip{background:#1b2233;color:var(--dim);border:1px solid var(--line);border-radius:9px;
   padding:6px 12px;cursor:pointer;font:inherit}
 .chip.active{color:var(--text);border-color:var(--acc);background:#1d2842}
 .chip b{color:var(--acc);margin-left:2px}
 #cnt{color:var(--dim);font-size:12px;margin-left:auto}
</style></head>
<body>
<header><h1>MultiSight3D · 资源模型画廊</h1>
<span>外部成品模型在线展示 · 点击卡片进入单个模型的单文件 demo</span></header>
<div class="bar">
 <input id="q" type="search" placeholder="🔍 搜索模型名称…" autocomplete="off"/>
 <div class="chips">__CHIPS__</div>
 <span id="cnt"></span>
</div>
<div class="grid" id="grid">__GRID__</div>
<script>
(function(){
  const cards=[...document.querySelectorAll('#grid .card')];
  const q=document.getElementById('q'), cnt=document.getElementById('cnt');
  let filter='';
  function apply(){
    const kw=(q.value||'').trim().toLowerCase(); let shown=0;
    cards.forEach(c=>{
      const okT=!filter||c.dataset.t===filter;
      const okQ=!kw||(c.dataset.name||'').includes(kw)||(c.textContent||'').toLowerCase().includes(kw);
      const vis=okT&&okQ; c.style.display=vis?'':'none'; if(vis)shown++;
    });
    cnt.textContent='显示 '+shown+' / '+cards.length+' 个';
  }
  document.querySelectorAll('.chip').forEach(ch=>ch.onclick=()=>{
    document.querySelectorAll('.chip').forEach(x=>x.classList.remove('active'));
    ch.classList.add('active'); filter=ch.dataset.f; apply();
  });
  q.addEventListener('input',apply);
  apply();
})();
</script>
</body></html>"""
