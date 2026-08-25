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

  /// BUILD 45 CANDIDATE — pin the ENTIRE token→durations path (BERT +
  /// durationEncoder + LSTM + proj + sigmoid/round) to the CPU stream.
  /// A2 fix: arm-D adjudication proved every token-identical chunk gets
  /// deterministically different durations on the device GPU (−56%..+193%);
  /// Mac CPU == Mac GPU sample counts prove the CPU reference is the model's
  /// true output. See KokoroTTS.generateAudio.
  public static nonisolated(unsafe) var cpuDurationHead = false

  /// BUILD 45 CANDIDATE (branch β) — pin the decoder's source-STFT and
  /// inverse-STFT to the CPU stream. Arm-B falsified the asStrided theory;
  /// spike geography (frame-quantised bursts) points at the tiny
  /// nFft=20 FFT/overlap-add kernels — the mlx #2205 wrong-kernel class on
  /// A-series. Tiny FFTs: expected performance-neutral. See Generator.
  public static nonisolated(unsafe) var cpuDecoderStft = false
}
