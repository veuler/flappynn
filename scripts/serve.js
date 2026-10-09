import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createLearningService } from './learning-service.js';
import { createRlService } from './rl-service.js';

const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8' };
export function serveDirectory(root, port, handler = null) {
  const server = http.createServer(async (request, response) => {
    try {
      const pathname = new URL(request.url, 'http://localhost').pathname;
      if (handler && await handler(request, response, pathname)) return;
    } catch { response.writeHead(400).end('Bad request'); return; }
    if (!['GET', 'HEAD'].includes(request.method)) {
      response.writeHead(405, { Allow: 'GET, HEAD' }).end();
      return;
    }
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const filename = path.resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`);
      const relative = path.relative(root, filename);
      if (relative.startsWith('..') || path.isAbsolute(relative)) {
        response.writeHead(403).end('Forbidden');
        return;
      }
      const body = await readFile(filename);
      response.writeHead(200, { 'Content-Type': types[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
      response.end(request.method === 'HEAD' ? undefined : body);
    } catch (error) {
      const malformed = error instanceof URIError;
      response.writeHead(malformed ? 400 : 404).end(malformed ? 'Bad request' : 'Not found');
    }
  });
  server.on('error', (error) => {
    console.error(error.code === 'EADDRINUSE'
      ? `Port ${port} is already in use. Choose another PORT or stop the existing server.`
      : error.message);
    process.exitCode = 1;
  });
  server.listen(port, '127.0.0.1', () => console.log(`Neural Flappy: http://127.0.0.1:${port}`));
  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const services = [createLearningService(root, 'speed-516'), createLearningService(root, 'lookahead-616'), createRlService(root)];
  serveDirectory(fileURLToPath(new URL('../src/', import.meta.url)), Number(process.env.PORT || 3030),
    async (request, response, pathname) => {
      if (pathname === '/api/learning' || pathname.startsWith('/api/learning/')) {
        response.writeHead(410, {'Content-Type':'application/json'}).end(JSON.stringify({error:'This lab has been retired. Use the 6 → 16 → 1 Lab.'}));
        return true;
      }
      for (const service of services) if (await service(request, response, pathname)) return true;
      return false;
    });
}
