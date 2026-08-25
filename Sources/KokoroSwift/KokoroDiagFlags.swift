//
//  Kokoro-tts-lib
//
//  BUILD 44→46 (2026-08-25) — engine flags that live below KokoroTTS's
//  instance surface. Build-46 shape: E2 barriers and the Swift duration head
//  are the PRODUCTION DEFAULTS; every falsified hunt lever (Z, F, E, PD, G,
//  token-P) has been removed — see BUILD44/46-DESIGN in the project notes
//  for the full falsification record.

import Foundation

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
  /// PRODUCTION DEFAULT ON: restores PyTorch-truth pacing on device, where
  /// the MLX duration path diverges by −56%..+193% frames. A/B-able OFF.
  public static nonisolated(unsafe) var swiftDurationHead = true
}
