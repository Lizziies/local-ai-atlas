// Tiny in-memory fixed-window rate limiter (per client IP). Good enough for a single Render
// instance; use a shared store if the service is ever scaled horizontally.
export function rateLimit({ windowMs = 60_000, max = 60, name = 'api' } = {}) {
  const hits = new Map();
  const timer = setInterval(() => {
    const now = Date.now();
    for (const [k, v] of hits) if (v.reset <= now) hits.delete(k);
  }, windowMs);
  timer.unref?.();
  const mw = (req, res, next) => {
    const key = req.ip || req.socket?.remoteAddress || 'unknown';
    const now = Date.now();
    let h = hits.get(key);
    if (!h || h.reset <= now) { h = { count: 0, reset: now + windowMs }; hits.set(key, h); }
    h.count++;
    res.set('RateLimit-Limit', String(max));
    res.set('RateLimit-Remaining', String(Math.max(0, max - h.count)));
    res.set('RateLimit-Reset', String(Math.ceil((h.reset - now) / 1000)));
    if (h.count > max) {
      res.set('Retry-After', String(Math.ceil((h.reset - now) / 1000)));
      return res.status(429).json({ error: 'rate_limited', message: `Too many requests to ${name}. Please slow down.` });
    }
    next();
  };
  mw.stop = () => clearInterval(timer);
  return mw;
}
