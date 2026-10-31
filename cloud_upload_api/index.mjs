import {createServer} from 'node:http';
import {handleUpload} from '@vercel/blob/client';
import {get} from '@vercel/blob';
import {pipeline} from 'node:stream/promises';

// Project-wide Vercel Authentication protects this endpoint and the Streamlit app.
const server = createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  if (request.method === 'GET' && url.pathname === '/api/files') {
    try {
      const key = url.searchParams.get('key') ?? '';
      if (!/^exports\/[a-zA-Z0-9_-]+\/[0-9a-f]{64}$/.test(key)) throw new Error('Invalid export');
      const result = await get(key, {access:'private'});
      if (!result) throw new Error('Export not found');
      const filename = (url.searchParams.get('name') ?? 'export.csv').replace(/[\r\n]/g, '_');
      response.setHeader('Content-Type','application/octet-stream');
      response.setHeader('Content-Disposition',"attachment; filename*=UTF-8''"+encodeURIComponent(filename));
      response.setHeader('Cache-Control','private, no-store');
      await pipeline(result.stream,response);
    } catch {
      if (!response.headersSent) response.writeHead(404).end('Export not found');
      else response.destroy();
    }
    return;
  }
  response.setHeader('Content-Type', 'application/json');
  if (request.method !== 'POST' || request.url?.split('?')[0] !== '/api/uploads') {
    response.writeHead(404).end(JSON.stringify({error:'Not found'}));
    return;
  }
  try {
    let content = '';
    for await (const chunk of request) {
      content += chunk.toString('utf8');
      if (content.length > 16384) throw new Error('Token request too large');
    }
    const result = await handleUpload({body: JSON.parse(content), request,
      onBeforeGenerateToken: async (pathname) => {
        if (!/^incoming\/[0-9a-f-]{36}\/[A-Za-z0-9_.-]+\.(csv|tsv|txt|gct)$/i.test(pathname)) {
          throw new Error('Unsupported upload path');
        }
        return {maximumSizeInBytes: 500 * 1024 * 1024,
          validUntil: Date.now() + 30 * 60 * 1000,
          addRandomSuffix: false, allowOverwrite: false};
      },
    });
    response.end(JSON.stringify(result));
  } catch {
    response.writeHead(400).end(JSON.stringify({error:'Unable to authorize upload'}));
  }
});
server.listen(Number(process.env.PORT ?? 80), '0.0.0.0');
