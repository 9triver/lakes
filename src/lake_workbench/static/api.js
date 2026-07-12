async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || payload.message || `HTTP ${response.status}`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

export function fetchJson(url) {
  return requestJson(url);
}

export function postJson(url, payload) {
  return requestJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function patchJson(url, payload) {
  return requestJson(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function deleteJson(url) {
  return requestJson(url, { method: "DELETE" });
}
