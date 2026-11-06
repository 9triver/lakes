import { ApiError } from "../../api/client";

export async function streamJson<T>(path: string, onEvent: (event: T) => void, signal: AbortSignal): Promise<void> {
  const response = await fetch(path, { signal });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new ApiError(response.status, payload.error || payload.message || `HTTP ${response.status}`);
  }
  if (!response.body) throw new Error("浏览器未提供结果数据流，请刷新页面后重试");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  const consume = () => {
    let end: number;
    while ((end = pending.indexOf("\n")) >= 0) {
      const line = pending.slice(0, end).trim();
      pending = pending.slice(end + 1);
      if (line) onEvent(JSON.parse(line) as T);
    }
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      consume();
    }
    pending += decoder.decode();
    consume();
    if (pending.trim()) onEvent(JSON.parse(pending) as T);
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
