//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXFFT
import MLXNN

// Hanning window implementation to replace np.hanning
func hanning(length: Int) -> MLXArray {
  if length == 1 {
    return MLXArray(1.0)
  }

  let n = MLXArray(Array(stride(from: Float(1 - length), to: Float(length), by: 2.0)))
  let factor = .pi / Float(length - 1)
  return 0.5 + 0.5 * cos(n * factor)
}

// Unwrap implementation to replace np.unwrap
func unwrap(p: MLXArray) -> MLXArray {
  let period: Float = 2.0 * .pi
  let discont: Float = period / 2.0

  let pDiff1 = p[0..., 0 ..< p.shape[1] - 1]
  let pDiff2 = p[0..., 1 ..< p.shape[1]]

  let pDiff = pDiff2 - pDiff1

  let intervalHigh: Float = period / 2.0
  let intervalLow: Float = -intervalHigh

  var pDiffMod = pDiff - intervalLow
  pDiffMod = (((pDiffMod % period) + period) % period) + intervalLow

  let ddSignArray = MLX.where(pDiff .> 0, intervalHigh, pDiffMod)

  pDiffMod = MLX.where(pDiffMod .== intervalLow, ddSignArray, pDiffMod)

  var phCorrect = pDiffMod - pDiff
  phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)

  return MLX.concatenated([p[0..., 0 ..< 1], p[0..., 1...] + phCorrect.cumsum(axis: 1)], axis: 1)
}

func mlxStft(
  x: MLXArray,
  nFft: Int = 800,
  hopLength: Int? = nil,
  winLength: Int? = nil,
  window: Any = "hann",
  center: Bool = true,
  padMode: String = "reflect"
) -> MLXArray {
  let hopLen = hopLength ?? nFft / 4
  let winLen = winLength ?? nFft

  var w: MLXArray
  if let windowStr = window as? String {
    if windowStr.lowercased() == "hann" {
      w = hanning(length: winLen + 1)[0 ..< winLen]
    } else {
      fatalError("Only hanning is supported for window, not \(windowStr)")
    }
  } else if let windowArray = window as? MLXArray {
    w = windowArray
  } else {
    fatalError("Window must be a string or MLXArray")
  }

  if w.shape[0] < nFft {
    let padSize = nFft - w.shape[0]
    w = MLX.concatenated([w, MLXArray.zeros([padSize])], axis: 0)
  }

  func pad(_ x: MLXArray, padding: Int, padMode: String = "reflect") -> MLXArray {
    if padMode == "constant" {
      return MLX.padded(x, width: [padding, padding])
    } else if padMode == "reflect" {
      let prefix = x[1 ..< padding + 1][.stride(by: -1)]
      let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
      return MLX.concatenated([prefix, x, suffix])
    } else {
      fatalError("Invalid pad mode \(padMode)")
    }
  }

  var xArray = x

  if center {
    xArray = pad(xArray, padding: nFft / 2, padMode: padMode)
  }

  let numFrames = 1 + (xArray.shape[0] - nFft) / hopLen
  if numFrames <= 0 {
    fatalError("Input is too short")
  }

  let frames = stftFrames(
    xArray, numFrames: numFrames, nFft: nFft, hopLen: hopLen,
    safe: KokoroDiagFlags.safeFraming)

  let spec = MLXFFT.rfft(frames * w)
  return spec.transposed(1, 0)
}

// BUILD 44 AUDIT (2026-08-25): the framing arithmetic is provably in-bounds —
// floor((len-nFft)/hop) guarantees (numFrames-1)*hop + nFft <= len, and with
// center padding len >= nFft always. The residual hazard is the asStrided
// CONTIGUITY CONTRACT: xArray is a lazily-concatenated reflect-pad built from
// negative-stride slice views (see pad() in mlxStft), and MLX.asStrided reads
// the input's flat buffer assuming row-contiguity. If that assumption is
// violated on the iOS Metal allocator path (recycled/donated buffers), reads
// land in garbage (stochastic spikes) or unmapped pages (SIGSEGV) — the
// mlx-swift #121 class. `safe` framing removes the assumption entirely.
//
// SAFE PATH IS O(1) GRAPH NODES: this STFT runs at SAMPLE rate (nFft=20,
// hop=5), so per-frame slicing would fan out ~len/hop ops (carlos ≈ 93k) —
// a hang on device. Because nFft % hop == 0, framing decomposes exactly:
// reshape x into hop-sized rows R[j] = x[j*hop ..< (j+1)*hop]; frame i is
// concat(R[i], R[i+1], …, R[i+ratio-1]). Exact-fit bound:
// (numFrames-1)*hop + nFft == (numFrames+ratio-1)*hop <= len — the audit
// bound verbatim, so the reshape slice always fits with zero padding.
// Per-frame slice-stack survives only as the nFft % hop != 0 fallback
// (dead for the current config).
func stftFrames(_ xArray: MLXArray, numFrames: Int, nFft: Int, hopLen: Int, safe: Bool) -> MLXArray {
  guard safe else {
    return MLX.asStrided(xArray, [numFrames, nFft], strides: [hopLen, 1])
  }
  if nFft % hopLen == 0 {
    let ratio = nFft / hopLen
    let usable = (numFrames + ratio - 1) * hopLen
    precondition(usable <= xArray.shape[0], "stftFrames: exact-fit bound violated")
    let r = xArray[0 ..< usable].reshaped([numFrames + ratio - 1, hopLen])
    return MLX.concatenated((0 ..< ratio).map { k in r[k ..< (k + numFrames)] }, axis: 1)
  }
  // Fallback (dead for current config): explicit per-frame slices.
  return MLX.stacked(
    (0 ..< numFrames).map { xArray[($0 * hopLen) ..< ($0 * hopLen + nFft)] },
    axis: 0
  )
}

/// BUILD 44 — bit-equality self-test for the safe framing path. Builds a
/// deterministic ramp of `sampleCount` samples, frames it both ways, and
/// returns the max abs difference (must be exactly 0).
public enum KokoroFramingSelfTest {
  public static func maxAbsDiff(sampleCount: Int, nFft: Int = 20, hopLen: Int = 5) -> Float {
    let x = MLXArray(0 ..< sampleCount).asType(.float32) * 0.001
    let numFrames = 1 + (sampleCount - nFft) / hopLen
    guard numFrames > 0 else { return -1 }
    let a = stftFrames(x, numFrames: numFrames, nFft: nFft, hopLen: hopLen, safe: false)
    let b = stftFrames(x, numFrames: numFrames, nFft: nFft, hopLen: hopLen, safe: true)
    return MLX.abs(a - b).max().item()
  }
}

func mlxIstft(
  x: MLXArray,
  hopLength: Int? = nil,
  winLength: Int? = nil,
  window: Any = "hann"
) -> MLXArray {
  let winLen = winLength ?? ((x.shape[1] - 1) * 2)
  let hopLen = hopLength ?? (winLen / 4)

  var w: MLXArray
  if let windowStr = window as? String {
    if windowStr.lowercased() == "hann" {
      w = hanning(length: winLen + 1)[0 ..< winLen]
    } else {
      fatalError("Only hanning window is supported")
    }
  } else if let windowArray = window as? MLXArray {
    w = windowArray
  } else {
    fatalError("Window must be a string or MLXArray")
  }

  if w.shape[0] < winLen {
    w = MLX.concatenated([w, MLXArray.zeros([winLen - w.shape[0]])], axis: 0)
  }

  let xTransposed = x.transposed(1, 0)
  let t = (xTransposed.shape[0] - 1) * hopLen + winLen
  let windowModLen = 20 / 5

  let wSquared = w * w
  let totalWsquared = MLX.concatenated(Array(repeating: wSquared, count: t / winLen))

  let output = MLXFFT.irfft(xTransposed, axis: 1) * w

  var outputs: [MLXArray] = []
  var windowSums: [MLXArray] = []

  for i in 0 ..< windowModLen {
    let outputStride = output[.stride(from: i, by: windowModLen), .ellipsis].reshaped([-1])
    let windowSumArray = totalWsquared[0 ..< outputStride.shape[0]]

    outputs.append(MLX.concatenated([
      MLXArray.zeros([i * hopLen]),
      outputStride,
      MLXArray.zeros([max(0, t - i * hopLen - outputStride.shape[0])]),
    ]))

    windowSums.append(MLX.concatenated([
      MLXArray.zeros([i * hopLen]),
      windowSumArray,
      MLXArray.zeros([max(0, t - i * hopLen - windowSumArray.shape[0])]),
    ]))
  }

  var reconstructed = outputs[0]
  var windowSum = windowSums[0]
  for i in 1 ..< windowModLen {
    reconstructed += outputs[i]
    windowSum += windowSums[i]
  }

  reconstructed =
    reconstructed[winLen / 2 ..< (reconstructed.shape[0] - winLen / 2)] /
    windowSum[winLen / 2 ..< (reconstructed.shape[0] - winLen / 2)]

  return reconstructed
}

class MLXSTFT {
  let filterLength: Int
  let hopLength: Int
  let winLength: Int
  let window: String

  var magnitude: MLXArray?
  var phase: MLXArray?

  init(filterLength: Int = 800, hopLength: Int = 200, winLength: Int = 800, window: String = "hann") {
    self.filterLength = filterLength
    self.hopLength = hopLength
    self.winLength = winLength
    self.window = window
  }

  func transform(inputData: MLXArray) -> (MLXArray, MLXArray) {
    var audioArray = inputData
    if audioArray.ndim == 1 {
      audioArray = audioArray.expandedDimensions(axis: 0)
    }

    var magnitudes: [MLXArray] = []
    var phases: [MLXArray] = []

    for batchIdx in 0 ..< audioArray.shape[0] {
      // Compute STFT
      let stft = mlxStft(
        x: audioArray[batchIdx],
        nFft: filterLength,
        hopLength: hopLength,
        winLength: winLength,
        window: window,
        center: true,
        padMode: "reflect"
      )

      let magnitude = MLX.abs(stft)

      // Replaces np.angle()
      let phase = MLX.atan2(stft.imaginaryPart(), stft.realPart())

      magnitudes.append(magnitude)
      phases.append(phase)
    }

    let magnitudesStacked = MLX.stacked(magnitudes, axis: 0)
    let phasesStacked = MLX.stacked(phases, axis: 0)

    return (magnitudesStacked, phasesStacked)
  }

  func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
    var reconstructed: [MLXArray] = []

    for batchIdx in 0 ..< magnitude.shape[0] {
      let phaseCont = unwrap(p: phase[batchIdx])

      // Combine magnitude and phase
      let stft = magnitude[batchIdx] * MLX.exp(MLXArray(real: 0, imaginary: 1) * phaseCont)

      // Inverse STFT
      let audio = mlxIstft(
        x: stft,
        hopLength: hopLength,
        winLength: winLength,
        window: window
      )
      reconstructed.append(audio)
    }

    let reconstructedStacked = MLX.stacked(reconstructed, axis: 0)
    return reconstructedStacked.expandedDimensions(axis: 1)
  }

  func callAsFunction(inputData: MLXArray) -> MLXArray {
    let (mag, ph) = transform(inputData: inputData)
    magnitude = mag
    phase = ph
    let reconstruction = inverse(magnitude: mag, phase: ph)
    return reconstruction.expandedDimensions(axis: -2)
  }
}
