/* MultiSight3D WebUI 壳：上传 → SSE 进度时间线 → GLB 预览 → 产物下载。 */
const $ = (s) => document.querySelector(s);
const STAGES = ['preprocess', 'sfm', 'mvs', 'mesh', 'texture', 'report'];
const LABEL = { preprocess: '预处理', sfm: 'SfM 稀疏', mvs: 'MVS 稠密', mesh: '网格', texture: '纹理', report: '报告' };

let picked = [];
let current = null;      // 当前任务 id
let es = null;           // EventSource
let viewerState = null;  // three.js 实例，切换任务时销毁

/* ---------- 照片选择 ---------- */
const drop = $('#drop'), filesInput = $('#files'), submitBtn = $('#submit');
drop.onclick = () => filesInput.click();
filesInput.onchange = () => setFiles([...filesInput.files]);
['dragover', 'dragenter'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add('on'); }));
['dragleave', 'drop'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove('on'); }));
drop.addEventListener('drop', (ev) => setFiles([...ev.dataTransfer.files].filter((f) => /\.(jpe?g|png)$/i.test(f.name))));

function setFiles(list) {
  picked = list;
  $('#fnames').textContent = list.length ? `已选 ${list.length} 张` : '支持 jpg / png，环绕物体拍摄 15–150 张';
  submitBtn.disabled = !list.length;
}

submitBtn.onclick = async () => {
  submitBtn.disabled = true; submitBtn.textContent = '上传中…';
  try {
    const fd = new FormData();
    fd.append('preset', $('#preset').value);
    picked.forEach((f) => fd.append('photos', f));
    const res = await fetch('/api/tasks', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
    const { task_id } = await res.json();
    setFiles([]); filesInput.value = '';
    await refreshTasks();
    selectTask(task_id);
  } catch (err) {
    alert('上传失败：' + err.message);
  } finally {
    submitBtn.textContent = '上传并开始重建';
    submitBtn.disabled = !picked.length;
  }
};

/* ---------- 任务列表 ---------- */
async function refreshTasks() {
  const { tasks } = await (await fetch('/api/tasks')).json();
  const ul = $('#tasks');
  ul.innerHTML = '';
  if (!tasks.length) { ul.innerHTML = '<li style="cursor:default;color:var(--dim)">暂无任务</li>'; return; }
  tasks.forEach((t) => {
    const li = document.createElement('li');
    if (t.task_id === current) li.classList.add('sel');
    li.innerHTML = `<div class="id">${t.task_id}</div>
      <div style="margin-top:4px"><span class="pill ${t.status}">${t.status}</span>
      <span class="pill">${t.overall}%</span><span class="pill">${(t.params || {}).preset || '-'}</span></div>`;
    li.onclick = () => selectTask(t.task_id);
    ul.appendChild(li);
  });
}
setInterval(refreshTasks, 4000);
refreshTasks();

/* ---------- 进度面板 + SSE ---------- */
function buildStageRows() {
  const box = $('#stages'); box.innerHTML = '';
  STAGES.forEach((s) => {
    const row = document.createElement('div');
    row.className = 'stage';
    row.innerHTML = `<span>${LABEL[s]}</span><div class="bar" id="bar-${s}"><i></i></div><b id="pct-${s}" style="font-size:12px">0%</b>`;
    box.appendChild(row);
  });
}

function selectTask(id) {
  current = id; es && es.close();
  $('#cur').textContent = '· ' + id;
  $('#log').textContent = '';
  $('#dl').innerHTML = '';
  buildStageRows();
  showHint('正在建立实时连接…');
  const setBar = (s, pct, cls) => {
    const bar = $(`#bar-${s}`); if (!bar) return;
    bar.className = 'bar ' + (cls || '');
    bar.querySelector('i').style.width = pct + '%';
    $(`#pct-${s}`).textContent = pct + '%';
  };
  es = new EventSource(`/api/tasks/${id}/events`);
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.type === 'progress') {
      setBar(ev.stage, ev.percent);
      if (ev.message) logLine(`[${LABEL[ev.stage] || ev.stage}] ${ev.message}`);
    } else if (ev.type === 'stage_done') {
      setBar(ev.stage, 100, 'ok');
    } else if (ev.type === 'stage_failed') {
      setBar(ev.stage, parseInt($(`#pct-${ev.stage}`).textContent) || 0, 'bad');
      logLine(`✘ ${ev.stage} 失败：${ev.error}`);
    } else if (ev.type === 'end') {
      es.close();
      logLine(ev.ok ? '✔ 管线完成' : '❌ 管线中断（已完成产物保留，可 --resume 续跑）');
      if (ev.ok) loadTaskArtifacts(id);
    }
  };
  es.onerror = () => { es.close(); };
  refreshTasks();
}

function logLine(text) {
  const el = $('#log');
  el.textContent += (el.textContent ? '\n' : '') + text;
  el.scrollTop = el.scrollHeight;
}

/* ---------- 产物下载 + 预览 ---------- */
async function loadTaskArtifacts(id) {
  const t = await (await fetch(`/api/tasks/${id}`)).json();
  const dl = $('#dl'); dl.innerHTML = '';
  (t.artifacts || []).filter((p) => /\.(glb|obj|ply|png|json)$/i.test(p)).forEach((p) => {
    const a = document.createElement('a');
    a.href = `/api/tasks/${id}/file?path=${encodeURIComponent(p)}`;
    a.download = p.split('/').pop();
    a.textContent = '⬇ ' + p;
    dl.appendChild(a);
  });
  if ((t.artifacts || []).includes('texture/model.glb')) loadViewer(`/api/tasks/${id}/file?path=texture/model.glb`);
}

function showHint(html) {
  const box = $('#viewer');
  let h = $('#vhint');
  if (!h) { h = document.createElement('div'); h.className = 'hint'; h.id = 'vhint'; box.appendChild(h); }
  h.style.display = 'grid'; h.innerHTML = html;
}

async function loadViewer(url) {
  showHint('加载 three.js 与模型…');
  let THREE, GLTFLoader, OrbitControls;
  try {
    THREE = await import('three');
    ({ GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js'));
    ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));
  } catch {
    showHint('three.js 需联网加载（unpkg CDN）<br />离线环境请直接下载 GLB 后用本地查看器打开');
    return;
  }
  destroyViewer();
  const box = $('#viewer');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.setSize(box.clientWidth, box.clientHeight);
  box.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x08090d);
  const camera = new THREE.PerspectiveCamera(45, box.clientWidth / box.clientHeight, 0.05, 100);
  camera.position.set(2.2, 1.6, 2.6);
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const dir = new THREE.DirectionalLight(0xbfd4ff, 1.6); dir.position.set(3, 5, 2); scene.add(dir);
  const grid = new THREE.GridHelper(6, 24, 0x2a3550, 0x1a2236); scene.add(grid);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.autoRotate = true;

  new GLTFLoader().load(url, (gltf) => {
    gltf.scene.traverse((o) => {
      if (o.isMesh && !o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    });
    scene.add(gltf.scene);
    $('#vhint').style.display = 'none';
  }, undefined, () => showHint('模型加载失败'));

  let raf;
  (function loop() { raf = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
  const onResize = () => {
    renderer.setSize(box.clientWidth, box.clientHeight);
    camera.aspect = box.clientWidth / box.clientHeight; camera.updateProjectionMatrix();
  };
  addEventListener('resize', onResize);
  viewerState = { dispose: () => { cancelAnimationFrame(raf); removeEventListener('resize', onResize); controls.dispose(); renderer.dispose(); renderer.domElement.remove(); } };
}

function destroyViewer() {
  if (viewerState) { viewerState.dispose(); viewerState = null; }
}
