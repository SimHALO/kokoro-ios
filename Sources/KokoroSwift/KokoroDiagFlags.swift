//
//  Kokoro-tts-lib
//
//  BUILD 44→46 (2026-08-25) — engine flags that live below KokoroTTS's
//  instance surface. Build-46 shape: E2 barriers and the Swift duration head
//  are the PRODUCTION DEFAULTS; every falsified hunt lever (Z, F, E, PD, G,
//  token-P) has been removed — see BUILD44/46-DESIGN in the project notes
//  for the full falsification record.

import Foundation
import MLX

public enum KokoroDiagFlags {
  // nonisolated(unsafe): writes happen only from the SimhaloKokoroEngine
  // facade under its lock, before synthesis starts, on the single serial
  // synth queue — external synchronization per the Swift 6 escape hatch.

  /// E2 — materialisation at decoder-internal boundaries (post source-STFT,
  /// per upsample block group, around the inverse STFT).
  /// PRODUCTION DEFAULT ON: kills the big-lazy-graph corruption class on
  /// A-series devices (three consecutive clean runs on the worst-case row)
  /// at ~±4% device cost. A/B-able OFF via the bench.
  public static nonisolated(unsafe) var decoderBarrier = true

  /// SD — deterministic Swift duration head (see SwiftDurationHead.swift).
  /// BUILD 47: PRODUCTION DEFAULT OFF. The head is truth-exact on Mac
  /// (gate 1: integer-exact vs PyTorch, all chunks) but device gate 2
  /// FAILED decisively: device frames ≈ 40–52% of truth on hash-identical
  /// rows (dan 46 vs 116, carlos 128 vs 264) — the divergence lives in
  /// device-side BERT features UPSTREAM of the head, far beyond the
  /// ±1–2-frame acceptance band. Lever retained for the BERT-divergence
  /// hunt; Mac kbench forces it ON (truth-exact there). Device pacing
  /// stays on the MLX path until BERT is resolved.
  public static nonisolated(unsafe) var swiftDurationHead = false
}

// BUILD 49 (2026-08-25) — GENERATOR-INTERNAL TELEMETRY.
// Evidence that forced this: on device, every stage feeding the vocoder is
// within ~2x of Mac (text_enc 0.65-0.69, dur_features 1.06-1.17, aligned
// 1.16-1.48, f0 0.74-0.81, n 0.42-0.51, asr 0.64-0.88) and then `audio`
// explodes 4.8-9.6x. Spectral analysis of device WAVs agrees: pitch structure
// intact (harmonicity 0.73-0.96) but 76-91% of energy above 6 kHz, centroid
// ~8.4 kHz vs Mac ~0.6 kHz — periodic excitation with NO spectral shaping.
// The break is inside Generator.callAsFunction. These stats name the block.
//
// Collected only when KokoroTTS.collectStageStats is on (bench S lever), so
// production pays nothing. Same external-synchronization contract as the
// flags above: written on the single serial synth queue.
public extension KokoroDiagFlags {
  /// BUILD 50 — PRODUCTION DEFAULT ON. Runs Generator's final convolution on
  /// the CPU stream. The Metal variant for this layer's shape returns a
  /// range-collapsed result on A-series (device 0.12-0.16x Mac RMS, range
  /// [-4.1,+5.0] vs Mac [-36.6,+14.4]) which flattens the magnitude spectrum
  /// after exp() and destroys all formant structure. Mac is unaffected
  /// (same values either way); the flag exists so the bench can A/B it.
  nonisolated(unsafe) static var cpuPostConv = true

  /// BUILD 52 — run the ENTIRE synthesis on the CPU stream. Default ON as the
  /// decisive test of whether A-series Metal kernel divergence explains the
  /// whole fault (it already explains the vocoder). A/B-able from the bench.
  nonisolated(unsafe) static var cpuSynthesis = true

  nonisolated(unsafe) static var collectGenStats = false
  nonisolated(unsafe) static var genStats: [String: [Float]] = [:]

  /// Record [min, max, rms] for a generator-internal tensor. The eval is
  /// implicit in .item(); acceptable because this path is bench-only.
  static func genStat(_ name: String, _ x: MLXArray) {
    guard collectGenStats else { return }
    let f = x.asType(.float32)
    let mn: Float = f.min().item()
    let mx: Float = f.max().item()
    let rms: Float = MLX.sqrt(MLX.mean(f * f)).item()
    genStats[name] = [mn, mx, rms]
  }
}
