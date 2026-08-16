/** Peak envelopes read off the actual audio. The file is decoded once with the Web Audio
 *  API and reduced to N buckets, each the loudest sample in its slice, normalised to the
 *  loudest bucket — so silence reads as silence and a burst reads as a burst. Clips are
 *  capped at 90s, so decoding client-side costs a few tens of milliseconds and saves the
 *  API from having to serve peak data it does not store. */

import { useEffect, useState } from "react";

let ctx: AudioContext | null = null;

/** One context for the page: browsers cap how many a document may create, and
 *  decodeAudioData works on a suspended one. */
function audioContext() {
  const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) throw new Error("no AudioContext");
  if (!ctx) ctx = new Ctor();
  return ctx;
}

export async function audioPeaks(source: string | Blob, n: number): Promise<number[]> {
  let bytes: ArrayBuffer;
  if (source instanceof Blob) {
    bytes = await source.arrayBuffer();
  } else {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`audio unavailable (${res.status})`);
    bytes = await res.arrayBuffer();
  }
  // mono is enough for an envelope, and every source here is speech
  const pcm = (await audioContext().decodeAudioData(bytes)).getChannelData(0);
  const size = pcm.length / n;
  const out: number[] = [];
  let max = 0;
  for (let i = 0; i < n; i++) {
    const end = Math.min(pcm.length, Math.round((i + 1) * size));
    let peak = 0;
    for (let j = Math.round(i * size); j < end; j++) {
      const v = Math.abs(pcm[j]);
      if (v > peak) peak = v;
    }
    out.push(peak);
    if (peak > max) max = peak;
  }
  return max ? out.map((v) => v / max) : out;
}

// Promises, not results, so StrictMode's double-mount and two players on one URL share a
// single decode instead of racing two.
const cache = new Map<string, Promise<number[]>>();

export type PeaksStatus = "loading" | "ready" | "failed";

export function usePeaks(src: string, n: number) {
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [status, setStatus] = useState<PeaksStatus>("loading");

  useEffect(() => {
    const key = `${n}|${src}`;
    let pending = cache.get(key);
    if (!pending) {
      pending = audioPeaks(src, n);
      pending.catch(() => cache.delete(key)); // a transient fetch failure shouldn't be permanent
      cache.set(key, pending);
    }
    let live = true;
    setStatus("loading");
    setPeaks(null);
    pending.then(
      (p) => { if (live) { setPeaks(p); setStatus("ready"); } },
      () => { if (live) setStatus("failed"); },
    );
    return () => { live = false; };
  }, [src, n]);

  return { peaks, status };
}
