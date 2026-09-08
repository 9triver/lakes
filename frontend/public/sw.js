const CACHE_NAME = "lakes-basemap-v2";
const MAX_TILES = 5000;

function isBasemapTile(request) {
  if (request.method !== "GET") return false;
  const url = new URL(request.url);
  if (url.hostname === "tile.openstreetmap.de") {
    return url.pathname.endsWith(".png");
  }
  return url.hostname === "server.arcgisonline.com"
    && url.pathname.includes("/ArcGIS/rest/services/World_Imagery/MapServer/tile/")
    && url.pathname.endsWith(".png");
}

async function trimCache(cache) {
  const keys = await cache.keys();
  const excess = keys.length - MAX_TILES;
  for (let index = 0; index < excess; index += 1) {
    await cache.delete(keys[index]);
  }
}

async function removeOldCaches() {
  const keys = await caches.keys();
  await Promise.all(
    keys
      .filter((key) => key.startsWith("lakes-basemap-") && key !== CACHE_NAME)
      .map((key) => caches.delete(key)),
  );
}

async function cachedTile(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    // Cross-origin tiles may be opaque when the provider omits CORS headers.
    if (response.ok || response.type === "opaque") {
      await cache.put(request, response.clone());
      await trimCache(cache);
    }
    return response;
  } catch (error) {
    return Response.error();
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(Promise.all([removeOldCaches(), self.clients.claim()]));
});

self.addEventListener("fetch", (event) => {
  if (isBasemapTile(event.request)) {
    event.respondWith(cachedTile(event.request));
  }
});
