export async function getJson<T>(path: string): Promise<T> {
  return requestJson<T>(path);
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  return requestJson<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export async function patchJson<T>(path: string, body: unknown): Promise<T> {
  return requestJson<T>(path, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export async function deleteJson<T>(path: string): Promise<T> {
  return requestJson<T>(path, { method: "DELETE" });
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || payload.message || `HTTP ${response.status}`);
  }
  return payload as T;
}
