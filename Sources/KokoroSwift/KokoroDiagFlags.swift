//
//  Kokoro-tts-lib
//
//  BUILD 44→45 (2026-08-25) — corruption-hunt levers that live below
//  KokoroTTS's instance surface (free functions / deep module internals
//  consult these). Set via the SimhaloKokoroEngine facade.
//  Build-45 disposition: decoderBarrier is the PRODUCTION DEFAULT (ON);
//  the two CPU pins are bench diagnostics; Z (safeFraming) was falsified
//  on device (arm B: corruption at control rates) and is REMOVED.

import Foundation

public enum KokoroDiagFlags {
  // nonisolated(unsafe): writes happen only from the SimhaloKokoroEngine
  // facade under its lock, before synthesis starts, on the single serial
  // synth queue — external synchronization per the Swift 6 escape hatch.

  /// E2 — materialisation at decoder-INTERNAL boundaries (post source-STFT,
  /// per upsample block group, around the inverse STFT).
  /// BUILD 45: PRODUCTION DEFAULT ON. Arm C proved it kills the big-lazy-
  /// graph corruption class outright (carlos, worst row of the hunt: 0/0/0
  /// under ±0.93 — first clean run ever) at ~±4% device cost. The flag stays
  /// so the bench can A/B it OFF. See Generator.callAsFunction.
  public static nonisolated(unsafe) var decoderBarrier = true

  /// DIAGNOSTIC — pin the ENTIRE token→durations path (BERT + durationEncoder
  /// + LSTM + proj + sigmoid/round) to the CPU stream. NOT a fix: ground truth
  /// (upstream Kokoro PyTorch) matches the MAC-GPU durations sample-exactly on
  /// all comparable texts, and BOTH device-GPU and CPU backends diverge from
  /// it (dan: PyTorch/Mac-GPU 116 vs device-GPU 96 vs CPU 55 — the duration
  /// head is numerically fragile off the reference path). Retained so the
  /// device bench can measure whether CPU-durations move the corruption.
  /// The 46 pacing fix is a deterministic plain-Swift duration head.
  public static nonisolated(unsafe) var cpuDurationHead = false

  /// G (bench toggle) — pin the decoder's source-STFT and inverse-STFT to the
  /// CPU stream. Targeted probe for the small-shape residual that survives
  /// S/Z/E2 (frame-quantised spike bursts implicate the tiny nFft=20
  /// FFT/overlap-add kernels — the mlx #2205 wrong-kernel class on A-series).
  /// Mac-verified shape-neutral. See Generator.
  public static nonisolated(unsafe) var cpuDecoderStft = false
}
