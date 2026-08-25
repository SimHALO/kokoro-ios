//
//  Kokoro-tts-lib
//
//  BUILD 46 (2026-08-25) — deterministic Swift duration head.
//
//  WHY THIS EXISTS: the duration head (stacked biLSTMs → sigmoid-sum → round)
//  is an integer cliff that amplifies backend fp differences into whole-frame
//  errors. Ground truth (upstream Kokoro PyTorch) matches Mac-MLX-GPU
//  sample-exactly, but device-MLX-GPU and MLX-CPU both diverge (dan: 116
//  truth vs 96 device vs 55 CPU — same weights, same tokens). This file
//  reimplements the ENTIRE token→durations path (durationEncoder stack +
//  predictor LSTM + projection + rounding) in plain Swift + Accelerate with
//  a fixed, single-threaded evaluation order — the same answer on every
//  device, every run. BERT stays MLX (attention/LayerNorm are numerically
//  tame; the two-gate acceptance in BUILD46-DESIGN.md verifies empirically).
//
//  Architecture mirrored 1:1 from DurationEncoder.swift + KokoroTTS
//  predictDurations @ 603acf0:
//    x = concat(bertEncoded[t], style)                       [T, 640]
//    3 × { biLSTM(640 → 256/dir = 512) ; AdaLayerNorm(style) ; concat style }
//    predictorLSTM(640 → 512) ; durationProj(512 → 50)
//    dur[t] = clip(round(sum(sigmoid(logits[t])) / speed), min: 1)
//  Gate order i,f,g,o (PyTorch layout); AdaLN uses population variance,
//  eps 1e-5, (1 + gamma) * x̂ + beta with gamma = fc(s)[0..<d].
//  Batch is always 1 and sequences carry no interior padding, so the mask
//  operations in the MLX path are no-ops here (boundary zeros are REAL
//  tokens, same as the MLX path treats them).

import Accelerate
import Foundation
import MLX

final class SwiftDurationHead {
  private let dModel: Int      // 512
  private let styleDim: Int    // 128
  private let nLayers: Int     // 3
  private let maxDur: Int      // 50 (proj out dim, read from weights)

  private struct BiLSTM {
    // PyTorch layout: weight_ih [4H, in], weight_hh [4H, H], biases [4H].
    let wxF: [Float], whF: [Float], bF: [Float]   // bF = bias_ih + bias_hh
    let wxB: [Float], whB: [Float], bB: [Float]
    let inSize: Int
    let hidden: Int
  }

  private let encoderLSTMs: [BiLSTM]           // nLayers
  private let adaFCw: [[Float]], adaFCb: [[Float]]  // nLayers × ([2d×styleDim], [2d])
  private let predictorLSTM: BiLSTM
  private let projW: [Float], projB: [Float]   // [maxDur×dModel], [maxDur]

  /// FAILABLE (build-46 review hardening): SD ships default-ON, so a missing
  /// weight key must NEVER hard-crash voice loading. On any missing key the
  /// init logs once and returns nil; the synth path then falls back to the
  /// MLX duration path — worst case is wrong pacing with working voices,
  /// same degrade philosophy as the zero-guard.
  init?(weights: [String: MLXArray], nLayers: Int, dModel: Int, styleDim: Int) {
    self.nLayers = nLayers
    self.dModel = dModel
    self.styleDim = styleDim

    var missingKey: String?
    func arr(_ key: String) -> [Float] {
      if let w = weights[key] { return w.asType(.float32).asArray(Float.self) }
      if missingKey == nil { missingKey = key }
      return []
    }
    func lstm(_ prefix: String, inSize: Int, hidden: Int) -> BiLSTM {
      let bF = zip(arr("\(prefix).bias_ih_l0"), arr("\(prefix).bias_hh_l0")).map(+)
      let bB = zip(arr("\(prefix).bias_ih_l0_reverse"), arr("\(prefix).bias_hh_l0_reverse")).map(+)
      return BiLSTM(
        wxF: arr("\(prefix).weight_ih_l0"), whF: arr("\(prefix).weight_hh_l0"), bF: bF,
        wxB: arr("\(prefix).weight_ih_l0_reverse"), whB: arr("\(prefix).weight_hh_l0_reverse"), bB: bB,
        inSize: inSize, hidden: hidden)
    }

    var encs: [BiLSTM] = []
    var fcw: [[Float]] = []
    var fcb: [[Float]] = []
    for i in 0 ..< nLayers * 2 {
      if i % 2 == 0 {
        encs.append(lstm("predictor.text_encoder.lstms.\(i)", inSize: dModel + styleDim, hidden: dModel / 2))
      } else {
        fcw.append(arr("predictor.text_encoder.lstms.\(i).fc.weight"))
        fcb.append(arr("predictor.text_encoder.lstms.\(i).fc.bias"))
      }
    }
    encoderLSTMs = encs
    adaFCw = fcw
    adaFCb = fcb
    predictorLSTM = lstm("predictor.lstm", inSize: dModel + styleDim, hidden: dModel / 2)
    projW = arr("predictor.duration_proj.linear_layer.weight")
    projB = arr("predictor.duration_proj.linear_layer.bias")
    maxDur = projB.count

    if let key = missingKey {
      print("[SwiftDurationHead] missing weight \(key) — head unavailable, falling back to MLX duration path")
      return nil
    }
  }

  /// One direction of an LSTM over the sequence. Fixed evaluation order:
  /// xProj via a single sgemm, then a strictly sequential recurrence with
  /// sgemv — deterministic on every ARM core.
  private func lstmDirection(
    _ x: [Float], T: Int, w: (wx: [Float], wh: [Float], b: [Float]),
    inSize: Int, H: Int, reverse: Bool, out: inout [Float], outOffsetPerT: Int, outBase: Int
  ) {
    let G = 4 * H
    // xProj[T, 4H] = x[T, in] × wxᵀ + b
    var xProj = [Float](repeating: 0, count: T * G)
    x.withUnsafeBufferPointer { xp in
      w.wx.withUnsafeBufferPointer { wp in
        xProj.withUnsafeMutableBufferPointer { op in
          cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            Int32(T), Int32(G), Int32(inSize),
            1, xp.baseAddress, Int32(inSize),
            wp.baseAddress, Int32(inSize),
            0, op.baseAddress, Int32(G))
        }
      }
    }
    for t in 0 ..< T {
      for gIdx in 0 ..< G { xProj[t * G + gIdx] += w.b[gIdx] }
    }

    var h = [Float](repeating: 0, count: H)
    var c = [Float](repeating: 0, count: H)
    var ifgo = [Float](repeating: 0, count: G)
    let order = reverse ? Array(stride(from: T - 1, through: 0, by: -1)) : Array(0 ..< T)
    for t in order {
      // ifgo = xProj[t] + wh × h
      for gIdx in 0 ..< G { ifgo[gIdx] = xProj[t * G + gIdx] }
      w.wh.withUnsafeBufferPointer { wp in
        h.withUnsafeBufferPointer { hp in
          ifgo.withUnsafeMutableBufferPointer { op in
            cblas_sgemv(
              CblasRowMajor, CblasNoTrans, Int32(G), Int32(H),
              1, wp.baseAddress, Int32(H), hp.baseAddress, 1,
              1, op.baseAddress, 1)
          }
        }
      }
      for j in 0 ..< H {
        let i = 1 / (1 + expf(-ifgo[j]))
        let f = 1 / (1 + expf(-ifgo[H + j]))
        let g = tanhf(ifgo[2 * H + j])
        let o = 1 / (1 + expf(-ifgo[3 * H + j]))
        c[j] = f * c[j] + i * g
        h[j] = o * tanhf(c[j])
      }
      let base = outBase + t * outOffsetPerT
      for j in 0 ..< H { out[base + j] = h[j] }
    }
  }

  /// biLSTM over [T, inSize] → [T, 2H] (forward ‖ backward, PyTorch order).
  private func biLSTM(_ x: [Float], T: Int, l: BiLSTM) -> [Float] {
    let H = l.hidden
    var out = [Float](repeating: 0, count: T * 2 * H)
    lstmDirection(x, T: T, w: (l.wxF, l.whF, l.bF), inSize: l.inSize, H: H,
                  reverse: false, out: &out, outOffsetPerT: 2 * H, outBase: 0)
    lstmDirection(x, T: T, w: (l.wxB, l.whB, l.bB), inSize: l.inSize, H: H,
                  reverse: true, out: &out, outOffsetPerT: 2 * H, outBase: H)
    return out
  }

  /// AdaLayerNorm over [T, d]: LN(last axis, population variance, eps 1e-5)
  /// then (1 + gamma) * x̂ + beta with [gamma; beta] = fc(style).
  private func adaLayerNorm(_ x: [Float], T: Int, layer: Int, style: [Float]) -> [Float] {
    let d = dModel
    // h[2d] = fcW[2d, styleDim] × style + fcB
    var hVec = adaFCb[layer]
    adaFCw[layer].withUnsafeBufferPointer { wp in
      style.withUnsafeBufferPointer { sp in
        hVec.withUnsafeMutableBufferPointer { op in
          cblas_sgemv(
            CblasRowMajor, CblasNoTrans, Int32(2 * d), Int32(styleDim),
            1, wp.baseAddress, Int32(styleDim), sp.baseAddress, 1,
            1, op.baseAddress, 1)
        }
      }
    }
    var out = [Float](repeating: 0, count: T * d)
    for t in 0 ..< T {
      var mean: Float = 0
      for j in 0 ..< d { mean += x[t * d + j] }
      mean /= Float(d)
      var variance: Float = 0
      for j in 0 ..< d { let dv = x[t * d + j] - mean; variance += dv * dv }
      variance /= Float(d)
      let inv = 1 / sqrtf(variance + 1e-5)
      for j in 0 ..< d {
        let norm = (x[t * d + j] - mean) * inv
        out[t * d + j] = (1 + hVec[j]) * norm + hVec[d + j]
      }
    }
    return out
  }

  private func concatStyle(_ x: [Float], T: Int, width: Int, style: [Float]) -> [Float] {
    var out = [Float](repeating: 0, count: T * (width + styleDim))
    let ow = width + styleDim
    for t in 0 ..< T {
      for j in 0 ..< width { out[t * ow + j] = x[t * width + j] }
      for j in 0 ..< styleDim { out[t * ow + width + j] = style[j] }
    }
    return out
  }

  /// Run the full head.
  /// - Parameters:
  ///   - features: bertEncoded host array, row-major [T × dModel]
  ///   - tokenCount: T (incl. boundary zeros)
  ///   - style: globalStyle [styleDim]
  ///   - speed: speech rate divisor
  /// - Returns: (durationFeatures row-major [T × (dModel+styleDim)] — the
  ///   exact tensor the MLX path feeds step 6 — and integer durations [T]).
  func run(features: [Float], tokenCount T: Int, style: [Float], speed: Float)
    -> (durationFeatures: [Float], durations: [Int])
  {
    precondition(features.count == T * dModel, "SwiftDurationHead: feature shape mismatch")
    precondition(style.count == styleDim, "SwiftDurationHead: style shape mismatch")

    var x = concatStyle(features, T: T, width: dModel, style: style)  // [T, 640]
    for l in 0 ..< nLayers {
      let lstmOut = biLSTM(x, T: T, l: encoderLSTMs[l])               // [T, 512]
      let normed = adaLayerNorm(lstmOut, T: T, layer: l, style: style)
      x = concatStyle(normed, T: T, width: dModel, style: style)      // [T, 640]
    }
    let durationFeatures = x

    let predOut = biLSTM(x, T: T, l: predictorLSTM)                   // [T, 512]
    // logits[T, maxDur] = predOut × projWᵀ + projB
    var logits = [Float](repeating: 0, count: T * maxDur)
    predOut.withUnsafeBufferPointer { xp in
      projW.withUnsafeBufferPointer { wp in
        logits.withUnsafeMutableBufferPointer { op in
          cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            Int32(T), Int32(maxDur), Int32(dModel),
            1, xp.baseAddress, Int32(dModel),
            wp.baseAddress, Int32(dModel),
            0, op.baseAddress, Int32(maxDur))
        }
      }
    }
    var durations = [Int](repeating: 1, count: T)
    for t in 0 ..< T {
      var s: Float = 0
      for k in 0 ..< maxDur {
        logits[t * maxDur + k] += projB[k]
        s += 1 / (1 + expf(-logits[t * maxDur + k]))
      }
      durations[t] = max(1, Int((s / speed).rounded()))
    }
    return (durationFeatures, durations)
  }
}
