//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class ReflectionPad1d: Module {
  let padding: IntOrPair

  init(padding: (Int, Int)) {
    self.padding = IntOrPair([padding.0, padding.1])
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // BUILD 46 (2026-08-25): TRUE reflection padding. This class zero-padded
    // since the port — a fidelity deviation vs upstream PyTorch that creates
    // an edge discontinuity the model never saw in training. The build-45
    // residual corruption concentrates exactly at buffer/conv edges (shorts
    // at chunk head, longs at tail) — an edge-discontinuity × device-kernel
    // INTERACTION (Mac zero-pads identically and stays clean, so this is a
    // stimulus removal + fidelity fix, not claimed as sole cause).
    // Reflection excludes the edge sample, matching torch ReflectionPad1d:
    // prefix = x[..., 1...l] reversed, suffix = x[..., n-1-r ..< n-1] reversed.
    let l = padding.first
    let r = padding.second
    let n = x.dim(-1)
    // Degenerate guard: reflection requires pad < n; fall back to zero-pad
    // rather than crash (never triggers for real synth lengths).
    guard l < n, r < n, l >= 0, r >= 0, n > 1 else {
      return MLX.padded(x, widths: [IntOrPair([0, 0]), IntOrPair([0, 0]), padding])
    }
    var parts: [MLXArray] = []
    if l > 0 {
      parts.append(x[.ellipsis, 1 ... l][.ellipsis, .stride(by: -1)])
    }
    parts.append(x)
    if r > 0 {
      parts.append(x[.ellipsis, (n - 1 - r) ..< (n - 1)][.ellipsis, .stride(by: -1)])
    }
    return parts.count == 1 ? x : MLX.concatenated(parts, axis: -1)
  }
}
