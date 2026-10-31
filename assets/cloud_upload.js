// Uploads go straight from the browser to private Supabase Storage through
// server-signed, single-use URLs. Files above the per-object limit are sent as
// parts followed by an index object, matching SupabaseStore on the server.
const PARTS_MAGIC = 'GTEX-PARTS-1\n';

function put(url, body, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('PUT', url);
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.setRequestHeader('x-upsert', 'true');
    request.upload.onprogress = (event) => onProgress(event.loaded);
    request.onload = () => (request.status >= 200 && request.status < 300 ? resolve() : reject(new Error(String(request.status))));
    request.onerror = () => reject(new Error('network'));
    request.send(body);
  });
}

export default function(component) {
  const {parentElement, data, setStateValue} = component;
  const input = parentElement.querySelector('input');
  const status = parentElement.querySelector('[role="status"]');
  const clear = parentElement.querySelector('button');
  const progress = parentElement.querySelector('progress');
  const limit = Math.round(data.max_size / 1024 / 1024);
  input.accept = data.accept;
  input.disabled = Boolean(data.disabled);
  status.textContent = data.current?.name ?? 'CSV, TSV, TXT, atau GCT - maksimum ' + limit + ' MB';
  clear.hidden = !data.current;
  clear.onclick = () => {input.value = ''; setStateValue('value', null);};
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    if (file.size > data.max_size) {
      status.textContent = 'Berkas melebihi batas ' + limit + ' MB.';
      return;
    }
    const target = data.target;
    input.disabled = true;
    clear.hidden = true;
    progress.hidden = false;
    progress.value = 0;
    status.textContent = 'Mengunggah ' + file.name;
    try {
      if (file.size <= data.part_size) {
        await put(target.url, file, (loaded) => {progress.value = 100 * loaded / Math.max(file.size, 1);});
      } else {
        const count = Math.ceil(file.size / data.part_size);
        for (let index = 0; index < count; index++) {
          const start = index * data.part_size;
          await put(target.parts[index], file.slice(start, start + data.part_size),
            (loaded) => {progress.value = 100 * (start + loaded) / file.size;});
        }
        await put(target.url, new Blob([PARTS_MAGIC + JSON.stringify({parts: count, size: file.size})]), () => {});
      }
      setStateValue('value', {id: target.id, sig: target.sig, name: file.name, size: file.size});
    } catch (error) {
      status.textContent = 'Unggah gagal. Periksa koneksi atau akses, lalu pilih berkas kembali.';
    } finally {
      input.disabled = Boolean(data.disabled);
      progress.hidden = true;
    }
  };
}
