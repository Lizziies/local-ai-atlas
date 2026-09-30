import { createApp } from './src/app.js';

const rawPort = process.env.PORT ?? '10000';
const PORT = Number(rawPort);
if (!Number.isInteger(PORT) || PORT < 0 || PORT > 65535) {
  console.error(`Invalid PORT "${rawPort}". Falling back is disabled so misconfiguration is visible.`);
  process.exit(1);
}

let app;
try {
  app = createApp();
} catch (e) {
  console.error('Failed to start Local AI Atlas:', e.message);
  process.exit(1);
}

// Render requires binding to 0.0.0.0 on $PORT.
const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`Local AI Atlas listening on 0.0.0.0:${server.address().port}`);
});

function shutdown(signal) {
  console.log(`${signal} received, shutting down`);
  app.locals.stop?.();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 5000).unref();
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
