//
//  Kokoro-tts-lib
//
//  BUILD 44 (2026-08-25) — corruption-hunt levers that live below KokoroTTS's
//  instance surface (free functions / deep module internals consult these).
//  Bench-driven; both default OFF; production behaviour identical unless a
//  lever is thrown. Set via the SimhaloKokoroEngine facade.

import Foundation

public enum KokoroDiagFlags {
  // nonisolated(unsafe): writes happen only from the SimhaloKokoroEngine
  // facade under its lock, before synthesis starts, on the single serial
  // synth queue — external synchronization per the Swift 6 escape hatch.

  /// Z — bypass MLX.asStrided in STFT framing: explicit slice-stack framing
  /// using only standard view primitives. Removes the asStrided contiguity
  /// assumption suspected in the device-only decoder corruption + SIGSEGV
  /// class (mlx-swift #121). See MLXSTFT.mlxStft.
  public static nonisolated(unsafe) var safeFraming = false

  /// E2 — force materialisation at decoder-INTERNAL boundaries (post
  /// source-STFT, per upsample block group, around the inverse STFT). The
  /// stage-level barrier was falsified by build-43 run-3: stats-ON synced
  /// every stage boundary and audio still corrupted — the defect is inside
  /// the decoder graph. See Generator.callAsFunction.
  public static nonisolated(unsafe) var decoderBarrier = false
}
