import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useTheme } from "@/lib/theme";
import {
  muted, ACCENT, GREEN, AMBER, AMBER_INK, VOICE_A, VOICE_B,
  MOOD_COLOR, MOOD_PCT, SENTIMENT_COLOR, type Mood,
} from "@/lib/ui";

gsap.registerPlugin(ScrollTrigger);

/* ── content ─────────────────────────────────────────────────────────────── */

const STAGES = [
  { n: "01", name: "Preprocess", note: "Decode any container, downmix to 16 kHz mono, normalise loudness, run VAD.", metric: "16 kHz · mono · ≤90 s" },
  { n: "02", name: "Transcribe", note: "faster-whisper large-v3-turbo on CTranslate2 — word timestamps and a logprob for every word.", metric: "large-v3-turbo · word-level" },
  { n: "03", name: "Diarize", note: "pyannote 3.1 segments the turns, snapped back to VAD edges and cleaned of micro-turns.", metric: "pyannote 3.1 · VAD-snapped" },
  { n: "04", name: "Reconcile", note: "Words are assigned to turns by overlap, low-confidence runs smoothed, then regrouped into utterances.", metric: "overlap assign · regroup" },
  { n: "05", name: "Embed", note: "ECAPA-TDNN takes each turn to a speaker vector; vectors live in pgvector beside the text.", metric: "ECAPA-TDNN · pgvector" },
  { n: "06", name: "Identify", note: "Match against enrolled profiles behind a margin gate — and abstain out loud when the margin is thin.", metric: "margin gate · abstention" },
  { n: "07", name: "Index", note: "Postgres FTS plus MiniLM vectors on an HNSW index, fused at query time by reciprocal rank.", metric: "FTS + HNSW · RRF" },
  { n: "08", name: "Review", note: "You correct, enrol, merge and export. Every correction is chained into the audit log.", metric: "SRT · VTT · RTTM · JSON" },
];

const FEATURES = [
  { t: "Transcription", d: "Word timestamps with per-word logprobs and automatic language detection, so low confidence is visible rather than hidden.", m: "faster-whisper" },
  { t: "Diarization", d: "Turn segmentation with min-turn and merge-gap cleanup, and an explicit warning wherever voices overlap.", m: "pyannote 3.1" },
  { t: "Speaker ID", d: "Embeddings matched against enrolled profiles, reliability-gated — the system says nothing rather than guessing.", m: "ECAPA + pgvector" },
  { t: "Clustering", d: "Unknown voices cluster across clips. Promote a cluster to a real profile in one click.", m: "cross-clip" },
  { t: "Hybrid search", d: "Keyword, semantic, or both fused by reciprocal rank — over every utterance in the corpus.", m: "FTS · MiniLM · RRF" },
  { t: "Sentiment & tone", d: "Multilingual XLM-R over the text, fused with the acoustic read, stored per utterance and rolled up per speaker.", m: "XLM-R + acoustic" },
  { t: "Live capture", d: "Microphone in, pause-aligned chunking, transcript and tone read as you speak.", m: "streaming" },
  { t: "Graph projection", d: "An optional Neo4j read model of clips, speakers, drivers, sessions and laps — rebuilt from Postgres, never authoritative.", m: "Neo4j · derived" },
];

const JOURNEY = [
  { k: "Step one", t: "Drop it in", d: "Audio or video, up to ninety seconds, one file or a whole batch. The job is queued and the batch survives a refresh.", m: "/upload" },
  { k: "Step two", t: "Watch it run", d: "Every stage reports as it lands. Waveform, turns and transcript fill in beneath a live status strip.", m: "/clips/:id" },
  { k: "Step three", t: "Correct it", d: "Rename a voice, enrol it, merge two profiles, reassign a turn. Nothing is silently overwritten — the audit log is hash-chained.", m: "/speakers" },
  { k: "Step four", t: "Ask the corpus", d: "Search across every utterance, or let the agent query it with read-only tools and cite the speech ids behind each claim.", m: "/ask" },
];

const OUTCOMES = [
  { t: "Named", d: "Above the threshold, with margin to spare.", c: VOICE_A, hatch: false },
  { t: "Suggested", d: "A plausible match, drawn italic, waiting on you.", c: VOICE_B, hatch: false },
  { t: "Unknown", d: "Judged — and nobody in the corpus matches.", c: "var(--color-neutral-600)", hatch: false },
  { t: "Abstained", d: "Declined to judge. Drawn hatched, never as a result.", c: AMBER, hatch: true },
];

const MOODS: { k: Mood; d: string }[] = [
  { k: "calm", d: "Baseline — steady pace, level tone." },
  { k: "tired", d: "Flattened energy, slower cadence." },
  { k: "stressed", d: "Elevated pitch and pace, tension in the voice." },
];

const SENTIMENTS = [
  { t: "Positive", k: "positive" as const, d: "Text reads favourably, and the acoustic signal agrees." },
  { t: "Negative", k: "negative" as const, d: "Text reads unfavourably, and the acoustic signal agrees." },
  { t: "Neutral", k: "neutral" as const, d: "No strong lean either way." },
];

const STATS = [
  { v: 8, s: "", l: "Pipeline stages" },
  { v: 0, s: "", l: "Outbound calls" },
  { v: 5, s: "", l: "Export formats" },
  { v: 100, s: "%", l: "On your hardware" },
];

/* ── small pieces ────────────────────────────────────────────────────────── */

function Corners() {
  return <>{["tl", "tr", "bl", "br"].map((c) => <i key={c} className={`corner ${c}`} />)}</>;
}

/** Hero waveform — a bar field that idles, and scrubs taller as the hero leaves. */
function Waveform() {
  const ref = useRef<SVGSVGElement>(null);
  const bars = useRef(Array.from({ length: 96 }, (_, i) =>
    0.18 + 0.82 * Math.abs(Math.sin(i * 0.37) * Math.cos(i * 0.11) + Math.sin(i * 1.7) * 0.35)));

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ctx = gsap.context(() => {
      gsap.from("rect", { scaleY: 0, duration: 1.1, ease: "power3.out", stagger: { each: 0.008, from: "center" } });
      gsap.to("rect", {
        scaleY: () => 0.45 + Math.random() * 0.9, duration: () => 0.9 + Math.random(),
        ease: "sine.inOut", repeat: -1, yoyo: true, stagger: { each: 0.02, from: "random" },
      });
    }, el);
    return () => ctx.revert();
  }, []);

  return (
    <svg ref={ref} viewBox="0 0 960 200" preserveAspectRatio="none"
         style={{ width: "100%", height: "100%", display: "block" }} aria-hidden>
      {bars.current.map((h, i) => (
        <rect key={i} x={i * 10} y={100 - (h * 96) / 2} width={4} height={h * 96}
              fill={i % 7 === 0 ? "var(--color-accent)" : "currentColor"}
              opacity={i % 7 === 0 ? 0.55 : 0.16}
              style={{ transformBox: "fill-box", transformOrigin: "center" }} />
      ))}
    </svg>
  );
}

/** Counts up the first time it scrolls into view. */
function Counter({ to, suffix }: { to: number; suffix: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const o = { v: 0 };
    const tw = gsap.to(o, {
      v: to, duration: 1.4, ease: "power2.out",
      scrollTrigger: { trigger: el, start: "top 88%" },
      onUpdate: () => { el.textContent = Math.round(o.v) + suffix; },
    });
    return () => { tw.scrollTrigger?.kill(); tw.kill(); };
  }, [to, suffix]);
  return <span ref={ref} className="lp-num">0{suffix}</span>;
}

/** Cursor-tracked glow on the blueprint cards. */
function useCardGlow() {
  return (e: React.MouseEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
  };
}

/** A control that leans toward the cursor and springs back. */
function Magnetic({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLSpanElement>(null);
  const move = (e: React.MouseEvent) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    gsap.to(el, {
      x: (e.clientX - r.left - r.width / 2) * 0.28,
      y: (e.clientY - r.top - r.height / 2) * 0.4,
      duration: 0.4, ease: "power3.out",
    });
  };
  const leave = () => gsap.to(ref.current, { x: 0, y: 0, duration: 0.6, ease: "elastic.out(1,0.4)" });
  return <span ref={ref} className="lp-mag" onMouseMove={move} onMouseLeave={leave}>{children}</span>;
}

/* ── page ────────────────────────────────────────────────────────────────── */

export default function Landing() {
  const root = useRef<HTMLDivElement>(null);
  const [theme, setTheme] = useTheme();
  const [stuck, setStuck] = useState(false);
  const [stage, setStage] = useState(0);
  const glow = useCardGlow();

  useEffect(() => {
    const onScroll = () => setStuck(window.scrollY > 40);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useLayoutEffect(() => {
    const ctx = gsap.context((self) => {
      const sel = self.selector!;
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      // Headline: each masked line slides up, one after the next.
      if (!reduce) {
        gsap.from(".lp-line > span", { yPercent: 115, duration: 1.05, ease: "power4.out", stagger: 0.09, delay: 0.1 });
        gsap.from(".lp-hero-fade", { autoAlpha: 0, y: 22, duration: 0.9, ease: "power3.out", stagger: 0.09, delay: 0.55 });
        gsap.to(".lp-hero-wave", {
          yPercent: -28, scaleY: 1.9, ease: "none",
          scrollTrigger: { trigger: ".lp-hero", start: "top top", end: "bottom top", scrub: 0.4 },
        });
      }

      // Everything marked .lp-reveal comes up as it enters.
      (sel(".lp-reveal") as HTMLElement[]).forEach((el) => {
        gsap.from(el, {
          y: reduce ? 0 : 34, autoAlpha: 0, duration: 0.8, ease: "power3.out",
          scrollTrigger: { trigger: el, start: "top 86%" },
        });
      });

      // Marquee: one loop, wrapped so it never runs out of track.
      if (!reduce) {
        gsap.to(".lp-marquee-track", {
          xPercent: -100, duration: 26, ease: "none", repeat: -1,
          modifiers: { xPercent: (v) => `${parseFloat(v) % 100}` },
        });
      }

      // Pipeline: pin the section and walk the active stage with scroll progress.
      const pipeEnd = `+=${STAGES.length * 320}`;
      ScrollTrigger.create({
        trigger: ".lp-pipeline", start: "top top", end: pipeEnd,
        pin: ".lp-pipeline-inner", scrub: true, invalidateOnRefresh: true,
        onUpdate: (t) => setStage(Math.min(STAGES.length - 1, Math.floor(t.progress * STAGES.length))),
      });
      // The connector draws itself in step with that same scroll.
      const flow = sel(".lp-flow-line")[0] as SVGPathElement | undefined;
      if (flow) {
        const len = flow.getTotalLength();
        gsap.set(flow, { strokeDasharray: len, strokeDashoffset: len });
        gsap.to(flow, {
          strokeDashoffset: 0, ease: "none",
          scrollTrigger: { trigger: ".lp-pipeline", start: "top top", end: pipeEnd, scrub: true },
        });
      }

      // Journey: four panels dragged sideways by vertical scroll.
      const jt = sel(".lp-journey-track")[0] as HTMLElement | undefined;
      if (jt && !reduce && window.innerWidth > 900) {
        gsap.to(jt, {
          x: () => -(jt.scrollWidth - window.innerWidth + 80), ease: "none",
          scrollTrigger: {
            trigger: ".lp-journey", start: "top top", pin: true, scrub: 0.6,
            end: () => `+=${jt.scrollWidth - window.innerWidth + 400}`, invalidateOnRefresh: true,
          },
        });
      }
    }, root);
    return () => ctx.revert();
  }, []);

  const active = STAGES[stage];

  return (
    <div className="lp" ref={root}>
      {/* ── nav ── */}
      <header className="lp-nav" data-stuck={stuck ? 1 : 0}>
        <Link to="/" style={{ color: "inherit", display: "grid", lineHeight: 1.02 }}>
          <span style={{ fontFamily: "var(--font-heading)", fontSize: 16, letterSpacing: "-.01em" }}>SPEAKER INTELLIGENCE</span>
          <span style={{ fontFamily: "var(--font-heading)", fontSize: 16, color: "var(--color-accent)" }}>WORKBENCH</span>
        </Link>
        <nav style={{ marginLeft: "auto", display: "flex", gap: 22, alignItems: "center" }}>
          <a className="lp-navlink lp-hide-sm" href="#capabilities">Capabilities</a>
          <a className="lp-navlink lp-hide-sm" href="#pipeline">Pipeline</a>
          <a className="lp-navlink lp-hide-sm" href="#journey">Journey</a>
          <button className="btn btn-ghost" style={{ fontSize: 12 }}
                  onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          <Link className="btn btn-primary" to="/library">Open workbench →</Link>
        </nav>
      </header>

      {/* ── hero ── */}
      <section className="lp-hero" style={{
        minHeight: "100svh", display: "flex", flexDirection: "column",
        justifyContent: "center", paddingTop: 108, overflow: "hidden",
      }}>
        <div className="lp-grid-bg" />
        <div className="lp-hero-wave" style={{
          position: "absolute", inset: "auto 0 0 0", height: "42vh",
          color: "var(--color-text)", zIndex: 0,
        }}>
          <Waveform />
        </div>

        <div className="lp-wrap" style={{ position: "relative", zIndex: 1 }}>
          <div className="lp-checker lp-hero-fade" style={{ marginBottom: 22 }} aria-hidden />

          <div className="lp-hero-fade" style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 22 }}>
            <span className="lp-dot" />
            <span className="kicker">On-premises · nothing leaves the box</span>
          </div>

          <h1 className="lp-h1">
            <span className="lp-line"><span>Every voice</span></span>
            <span className="lp-line"><span style={{ color: "var(--color-accent)" }}>accounted for.</span></span>
            <span className="lp-line"><span>Nothing guessed.</span></span>
          </h1>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 40, alignItems: "flex-end", marginTop: 38 }}>
            <p className="lp-lede lp-hero-fade" style={{ margin: 0, color: muted(72), flex: "1 1 420px" }}>
              Drop in a clip and the workbench runs the whole chain offline — preprocess, transcribe,
              diarize, reconcile, embed, identify, index — then hands you a transcript you can correct,
              search and export. Models sit on your disk. No audio and no text ever leaves the machine.
            </p>
            <div className="lp-hero-fade" style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <Magnetic><Link className="btn btn-primary" to="/upload" style={{ padding: "12px 22px", fontSize: 15 }}>Upload a clip</Link></Magnetic>
              <Magnetic><Link className="btn btn-secondary" to="/library" style={{ padding: "12px 22px", fontSize: 15 }}>Browse the library</Link></Magnetic>
            </div>
          </div>

          <div className="lp-hero-fade" style={{
            marginTop: 56, display: "grid", gap: 1, gridTemplateColumns: "repeat(auto-fit, minmax(min(160px, 100%), 1fr))",
            background: "var(--color-divider)", border: "1px solid var(--color-divider)",
          }}>
            {STATS.map((s) => (
              <div key={s.l} style={{ background: "var(--color-bg)", padding: "18px 20px" }}>
                <Counter to={s.v} suffix={s.s} />
                <div className="kicker" style={{ marginTop: 8 }}>{s.l}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── marquee ── */}
      <div className="lp-marquee">
        {[0, 1].map((k) => (
          <div className="lp-marquee-track" key={k} aria-hidden={k === 1}>
            {STAGES.map((s) => (
              <span key={s.name}>
                <b aria-hidden>🏁</b>&nbsp;&nbsp;{s.name}&nbsp;&nbsp;
                <span style={{ opacity: 0.55 }}>{s.metric}</span>
              </span>
            ))}
          </div>
        ))}
      </div>

      {/* ── capabilities ── */}
      <section id="capabilities" style={{ padding: "120px 0 40px", scrollMarginTop: 80 }}>
        <div className="lp-wrap">
          <div className="lp-reveal" style={{ display: "flex", gap: 40, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 46 }}>
            <div style={{ flex: "1 1 460px" }}>
              <span className="kicker">§ 01 — Capabilities</span>
              <h2 className="lp-h2" style={{ marginTop: 12 }}>A workbench,<br />not a black box.</h2>
            </div>
            <p className="lp-lede" style={{ flex: "1 1 380px", color: muted(66), margin: 0 }}>
              Every model is swappable from Settings, every pipeline knob is editable live, and every
              claim the system makes points back at the audio it came from.
            </p>
          </div>

          <div style={{ display: "grid", gap: 14, gridTemplateColumns: "repeat(auto-fit, minmax(min(268px, 100%), 1fr))" }}>
            {FEATURES.map((f, i) => (
              <article key={f.t} className="lp-card lp-reveal" onMouseMove={glow}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span className="lp-idx">{String(i + 1).padStart(2, "0")}</span>
                  <span className="mono" style={{ fontSize: 11, color: ACCENT }}>{f.m}</span>
                </div>
                <h3 style={{ fontSize: 22, margin: "16px 0 8px", textTransform: "uppercase", letterSpacing: "-.01em" }}>{f.t}</h3>
                <p style={{ margin: 0, fontSize: 14, color: muted(68) }}>{f.d}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* ── pipeline (pinned, scroll-scrubbed) ── */}
      <section id="pipeline" className="lp-pipeline" style={{ padding: "80px 0 0", scrollMarginTop: 80 }}>
        <div className="lp-pipeline-inner" style={{
          minHeight: "100svh", display: "flex", alignItems: "center", overflow: "hidden",
        }}>
          <div className="lp-grid-bg" />
          <div className="lp-wrap" style={{ width: "100%", position: "relative", zIndex: 1 }}>
            <div style={{ marginBottom: 34 }}>
              <span className="kicker">§ 02 — The pipeline</span>
              <h2 className="lp-h2" style={{ marginTop: 10 }}>Eight stages, in the open.</h2>
            </div>

            <div className="lp-pipe-grid">
              {/* stage list */}
              <div style={{ display: "grid", gap: 6 }}>
                {STAGES.map((s, i) => (
                  <div key={s.name} className="lp-stage" data-on={i === stage ? 1 : 0}>
                    <div style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
                      <span className="mono" style={{ fontSize: 11, color: i === stage ? ACCENT : muted(40) }}>{s.n}</span>
                      <span className="lp-stage-name" style={{ color: i <= stage ? "var(--color-text)" : muted(45) }}>{s.name}</span>
                      <span className="mono" style={{ marginLeft: "auto", fontSize: 10, color: muted(38) }}>
                        {i < stage ? "done" : i === stage ? "running" : "queued"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* connector */}
              <svg className="lp-flow lp-hide-sm" viewBox="0 0 74 400" preserveAspectRatio="none"
                   style={{ height: 400, width: 74 }} aria-hidden>
                <path d="M0 200 H30 V40 H74" stroke="var(--color-divider)" strokeWidth="1" />
                <path d="M0 200 H30 V360 H74" stroke="var(--color-divider)" strokeWidth="1" />
                <path className="lp-flow-line" d="M0 200 H30 V40 H74" stroke="var(--color-accent)" strokeWidth="2" />
                <circle r="3.5" fill="var(--color-accent)" cx="74" cy="40" />
              </svg>

              {/* detail panel */}
              <div className="blueprint" style={{ padding: "26px 28px", background: "var(--color-surface)", minHeight: 300 }}>
                <Corners />
                <div key={active.name} className="lp-swap">
                  <span className="mono" style={{ fontSize: 11, color: ACCENT }}>STAGE {active.n} / 08</span>
                  <h3 style={{ fontSize: "clamp(28px,3.4vw,48px)", margin: "10px 0 14px", textTransform: "uppercase" }}>{active.name}</h3>
                  <p style={{ fontSize: 16, color: muted(74), maxWidth: "52ch" }}>{active.note}</p>
                  <div className="hr" />
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
                    <span className="tag tag-outline mono" style={{ fontSize: 11 }}>{active.metric}</span>
                    <span className="mono" style={{ fontSize: 11, color: muted(45) }}>runs on your hardware · offline</span>
                  </div>
                  <div className="bar" style={{ marginTop: 22 }}>
                    <span style={{ width: `${((stage + 1) / STAGES.length) * 100}%`, background: ACCENT, transition: "width .4s ease" }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── journey (horizontal) ── */}
      <section id="journey" className="lp-journey" style={{
        minHeight: "100svh", display: "flex", flexDirection: "column",
        justifyContent: "center", overflow: "hidden", paddingBlock: 90, scrollMarginTop: 80,
      }}>
        <div className="lp-wrap" style={{ marginBottom: 40 }}>
          <span className="kicker">§ 03 — The journey</span>
          <h2 className="lp-h2" style={{ marginTop: 10 }}>Clip in, answer out.</h2>
        </div>
        <div className="lp-journey-track">
          {JOURNEY.map((j, i) => (
            <article key={j.t} className="lp-card" onMouseMove={glow}
                     style={{ width: "min(78vw, 460px)", flex: "none", minHeight: 380, display: "flex", flexDirection: "column" }}>
              <span className="kicker">{j.k}</span>
              <h3 style={{ fontSize: 34, margin: "14px 0 12px", textTransform: "uppercase" }}>{j.t}</h3>
              <p style={{ color: muted(70), fontSize: 15 }}>{j.d}</p>
              <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                <span className="mono" style={{ fontSize: 12, color: ACCENT }}>{j.m}</span>
                <span className="mono" style={{ fontSize: 42, lineHeight: 1, color: muted(14) }}>{String(i + 1).padStart(2, "0")}</span>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* ── four-state vocabulary ── */}
      <section style={{ padding: "110px 0" }}>
        <div className="lp-wrap">
          <div className="lp-reveal" style={{ display: "flex", gap: 40, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 40 }}>
            <div style={{ flex: "1 1 440px" }}>
              <span className="kicker">§ 04 — Saying nothing, on purpose</span>
              <h2 className="lp-h2" style={{ marginTop: 10 }}>Four states.<br />A refusal is one of them.</h2>
            </div>
            <p className="lp-lede" style={{ flex: "1 1 360px", color: muted(66), margin: 0 }}>
              Identification is reliability-gated. When the margin between the best two candidates is
              too thin, the workbench abstains — and abstention is drawn hatched, so it can never be
              mistaken for an answer.
            </p>
          </div>
          <div style={{ display: "grid", gap: 14, gridTemplateColumns: "repeat(auto-fit, minmax(min(220px, 100%), 1fr))" }}>
            {OUTCOMES.map((o) => (
              <div key={o.t} className="lp-reveal" style={{
                border: "1px solid var(--color-divider)", padding: "20px 20px 24px", background: "var(--color-bg)",
              }}>
                <div className={o.hatch ? "hatch" : ""} style={{ height: 8, background: o.hatch ? undefined : o.c, marginBottom: 18 }} />
                <div style={{ fontFamily: "var(--font-heading)", fontSize: 21, textTransform: "uppercase", color: o.hatch ? AMBER_INK : o.c }}>{o.t}</div>
                <p style={{ margin: "8px 0 0", fontSize: 14, color: muted(65) }}>{o.d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── sentiment & tone ── */}
      <section style={{ padding: "0 0 110px" }}>
        <div className="lp-wrap">
          <div className="lp-reveal" style={{ display: "flex", gap: 40, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 40 }}>
            <div style={{ flex: "1 1 440px" }}>
              <span className="kicker">§ 05 — Reading the room</span>
              <h2 className="lp-h2" style={{ marginTop: 10 }}>Two signals.<br />Fused, not averaged.</h2>
            </div>
            <p className="lp-lede" style={{ flex: "1 1 360px", color: muted(66), margin: 0 }}>
              Multilingual XLM-R reads the words; an acoustic model reads the voice underneath them.
              Both are stored per utterance and rolled up per speaker — so a flat transcript and a
              tense delivery are never collapsed into one number.
            </p>
          </div>

          <div style={{ display: "grid", gap: 34, gridTemplateColumns: "repeat(auto-fit, minmax(min(320px, 100%), 1fr))" }}>
            <div className="lp-card lp-reveal" onMouseMove={glow}>
              <span className="lp-idx">Acoustic — tone</span>
              <h3 style={{ fontSize: 20, margin: "12px 0 16px", textTransform: "uppercase" }}>How it was said</h3>
              <div style={{ display: "grid", gap: 16 }}>
                {MOODS.map((m) => (
                  <div key={m.k}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                      <span style={{ textTransform: "capitalize", color: MOOD_COLOR[m.k] }}>{m.k}</span>
                      <span className="mono" style={{ fontSize: 11, color: muted(45) }}>{MOOD_PCT[m.k]}</span>
                    </div>
                    <div className="bar"><span style={{ width: MOOD_PCT[m.k], background: MOOD_COLOR[m.k] }} /></div>
                    <p style={{ margin: "6px 0 0", fontSize: 12, color: muted(58) }}>{m.d}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="lp-card lp-reveal" onMouseMove={glow}>
              <span className="lp-idx">Text — sentiment</span>
              <h3 style={{ fontSize: 20, margin: "12px 0 16px", textTransform: "uppercase" }}>What was said</h3>
              <div style={{ display: "grid", gap: 14 }}>
                {SENTIMENTS.map((s) => (
                  <div key={s.k} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <span style={{ width: 8, height: 8, marginTop: 5, flex: "none", background: SENTIMENT_COLOR[s.k] }} />
                    <div>
                      <div style={{ fontSize: 14, color: SENTIMENT_COLOR[s.k] }}>{s.t}</div>
                      <p style={{ margin: "2px 0 0", fontSize: 12, color: muted(58) }}>{s.d}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="hr" style={{ margin: "18px 0 12px" }} />
              <span className="mono" style={{ fontSize: 11, color: muted(45) }}>a disagreement between the two is surfaced, not hidden</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── offline ── */}
      <section style={{ padding: "0 0 110px" }}>
        <div className="lp-wrap">
          <div className="blueprint lp-reveal" style={{ padding: "clamp(28px,5vw,64px)", background: "var(--color-surface)" }}>
            <Corners />
            <div style={{ display: "grid", gap: 40, gridTemplateColumns: "repeat(auto-fit, minmax(min(300px, 100%), 1fr))", alignItems: "center" }}>
              <div>
                <span className="kicker">§ 06 — Air-gapped by default</span>
                <h2 className="lp-h2" style={{ marginTop: 10, fontSize: "clamp(28px,3.6vw,50px)" }}>HF_HUB_OFFLINE=1</h2>
                <p style={{ color: muted(70), marginTop: 14, maxWidth: "48ch" }}>
                  Models are read from <span className="mono">MODEL_DIR</span> on disk. The core pipeline makes no
                  outbound call at all. Two features can reach the network — Race Radio and the agent layer —
                  and both stay off until you turn them on.
                </p>
              </div>
              <div style={{ display: "grid", gap: 1, background: "var(--color-divider)", border: "1px solid var(--color-divider)" }}>
                {([
                  ["Audio leaves the box", "never", GREEN],
                  ["Transcript leaves the box", "never", GREEN],
                  ["Race Radio (OpenF1)", "opt-in", AMBER_INK],
                  ["Agent layer (LLM_ENABLED)", "opt-in", AMBER_INK],
                  ["Audit log", "hash-chained", ACCENT],
                ] as const).map(([k, v, c]) => (
                  <div key={k} style={{
                    background: "var(--color-surface)", padding: "13px 16px",
                    display: "flex", justifyContent: "space-between", gap: 16,
                  }}>
                    <span style={{ fontSize: 14, color: muted(72) }}>{k}</span>
                    <span className="mono" style={{ fontSize: 12, color: c }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── footer cta ── */}
      <footer style={{ borderTop: "1px solid var(--color-divider)", padding: "90px 0 44px", position: "relative", overflow: "hidden" }}>
        <div className="lp-wrap" style={{ position: "relative", zIndex: 1 }}>
          <div className="lp-checker lp-reveal" style={{ marginBottom: 22, maxWidth: 220 }} aria-hidden />
          <h2 className="lp-h1 lp-reveal" style={{ fontSize: "clamp(40px,8vw,116px)" }}>Start listening.</h2>
          <div className="lp-reveal" style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 30 }}>
            <Magnetic><Link className="btn btn-primary" to="/upload" style={{ padding: "12px 24px", fontSize: 15 }}>Upload a clip</Link></Magnetic>
            <Magnetic><Link className="btn btn-secondary" to="/live" style={{ padding: "12px 24px", fontSize: 15 }}>Go live</Link></Magnetic>
            <Magnetic><Link className="btn btn-secondary" to="/ask" style={{ padding: "12px 24px", fontSize: 15 }}>Ask the corpus</Link></Magnetic>
          </div>
          <div className="hr" style={{ marginTop: 56 }} />
          <div style={{ display: "flex", justifyContent: "space-between", gap: 20, flexWrap: "wrap", fontSize: 12, color: muted(50) }}>
            <span className="mono">Speaker Intelligence Workbench · v2.4</span>
            <span className="mono">on-premises · offline · auditable</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
