import { upload } from '@vercel/blob/client';

export default function(component) {
  const {parentElement, data, setStateValue} = component;
  const input = parentElement.querySelector('input');
  const status = parentElement.querySelector('[role="status"]');
  const clear = parentElement.querySelector('button');
  const progress = parentElement.querySelector('progress');
  input.accept = data.accept;
  input.disabled = Boolean(data.disabled);
  status.textContent = data.current?.name ?? 'CSV, TSV, TXT, atau GCT - maksimum 500 MB';
  clear.hidden = !data.current;
  clear.onclick = () => {input.value = ''; setStateValue('value', null);};
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    if (file.size > 500 * 1024 * 1024) {
      status.textContent = 'Berkas melebihi batas 500 MB.';
      return;
    }
    input.disabled = true;
    clear.hidden = true;
    progress.hidden = false;
    status.textContent = 'Mengunggah ' + file.name;
    try {
      const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, '_');
      const pathname = 'incoming/' + crypto.randomUUID() + '/' + safeName;
      const result = await upload(pathname, file, {
        access: 'private', handleUploadUrl: '/api/uploads', multipart: file.size > 5 * 1024 * 1024,
        onUploadProgress: ({percentage}) => {progress.value = percentage;},
      });
      setStateValue('value', {pathname: result.pathname, name: file.name, size: file.size});
    } catch (error) {
      status.textContent = 'Unggah gagal. Periksa koneksi atau akses, lalu pilih berkas kembali.';
    } finally {
      input.disabled = Boolean(data.disabled);
      progress.hidden = true;
    }
  };
}
