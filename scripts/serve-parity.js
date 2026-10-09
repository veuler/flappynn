import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { serveDirectory } from './serve.js';

serveDirectory(fileURLToPath(new URL('../tests/browser/', import.meta.url)), Number(process.env.PARITY_PORT || 3032),
  async (request, response, pathname) => {
    if (pathname !== '/js/css-numbers.js') return false;
    if (!['GET', 'HEAD'].includes(request.method)) {
      response.writeHead(405, { Allow: 'GET, HEAD' }).end();
      return true;
    }
    const source = await readFile(new URL('../src/js/css-numbers.js', import.meta.url));
    response.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store' });
    response.end(request.method === 'HEAD' ? undefined : source);
    return true;
  });
