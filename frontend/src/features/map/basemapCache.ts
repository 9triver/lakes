import type { LoadFunction } from "ol/Tile";
import ImageTile from "ol/ImageTile";

const DATABASE_NAME = "lakes-basemap-cache-v2";
const STORE_NAME = "tiles";
const DATABASE_VERSION = 1;
const MAX_TILES = 5000;

interface CachedTile {
  url: string;
  blob: Blob;
  lastUsed: number;
}

const pendingLoads = new Map<string, Promise<Blob | null>>();

function canUseIndexedDb() {
  return typeof indexedDB !== "undefined";
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME, { keyPath: "url" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Unable to open basemap cache"));
  });
}

async function readTile(url: string): Promise<Blob | null> {
  if (!canUseIndexedDb()) return null;
  const database = await openDatabase();
  try {
    const tile = await new Promise<CachedTile | undefined>((resolve, reject) => {
      const request = database.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(url);
      request.onsuccess = () => resolve(request.result as CachedTile | undefined);
      request.onerror = () => reject(request.error);
    });
    if (!tile) return null;
    await touchTile(database, tile);
    return tile.blob;
  } finally {
    database.close();
  }
}

function touchTile(database: IDBDatabase, tile: CachedTile): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = database
      .transaction(STORE_NAME, "readwrite")
      .objectStore(STORE_NAME)
      .put({ ...tile, lastUsed: Date.now() });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

async function writeTile(url: string, blob: Blob) {
  if (!canUseIndexedDb()) return;
  const database = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const request = database
        .transaction(STORE_NAME, "readwrite")
        .objectStore(STORE_NAME)
        .put({ url, blob, lastUsed: Date.now() } satisfies CachedTile);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    const tiles = await new Promise<CachedTile[]>((resolve, reject) => {
      const request = database.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).getAll();
      request.onsuccess = () => resolve(request.result as CachedTile[]);
      request.onerror = () => reject(request.error);
    });
    tiles.sort((left, right) => left.lastUsed - right.lastUsed);
    const excess = tiles.length - MAX_TILES;
    if (excess <= 0) return;
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      const store = transaction.objectStore(STORE_NAME);
      for (const tile of tiles.slice(0, excess)) store.delete(tile.url);
    });
  } finally {
    database.close();
  }
}

async function fetchTile(url: string): Promise<Blob | null> {
  const existing = pendingLoads.get(url);
  if (existing) return existing;

  const load = (async () => {
    try {
      const cached = await readTile(url);
      if (cached) return cached;
      const response = await fetch(url, { mode: "cors" });
      if (!response.ok) return null;
      const blob = await response.blob();
      await writeTile(url, blob).catch(() => undefined);
      return blob;
    } catch {
      return null;
    }
  })();
  pendingLoads.set(url, load);
  try {
    return await load;
  } finally {
    pendingLoads.delete(url);
  }
}

function loadDirectly(image: HTMLImageElement, url: string) {
  image.src = url;
}

/**
 * IndexedDB fallback for HTTP IP access, where Service Workers are unavailable.
 * Failed CORS fetches still use OpenLayers' normal image loading path.
 */
export function createBasemapTileLoadFunction(): LoadFunction | undefined {
  if (typeof window === "undefined" || window.isSecureContext || !canUseIndexedDb()) return undefined;

  return (tile, url) => {
    if (!(tile instanceof ImageTile)) return;
    const image = tile.getImage();
    if (!(image instanceof HTMLImageElement)) return;
    void fetchTile(url).then((blob) => {
      if (!blob) {
        loadDirectly(image, url);
        return;
      }
      const objectUrl = URL.createObjectURL(blob);
      image.onload = () => URL.revokeObjectURL(objectUrl);
      image.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        loadDirectly(image, url);
      };
      image.src = objectUrl;
    });
  };
}
