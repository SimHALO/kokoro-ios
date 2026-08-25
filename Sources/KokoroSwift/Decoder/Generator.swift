//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Generator {
  let numKernels: Int
  let numUpsamples: Int
  let mSource: SourceModuleHnNSF
  let f0Upsample: Upsample
  let postNFFt: Int
  var noiseConvs: [Conv1dInference]
  var noiseRes: [AdaINResBlock1]
  var ups: [ConvWeighted]
  var resBlocks: [AdaINResBlock1]
  let convPost: ConvWeighted
  let reflectionPad: ReflectionPad1d
  let stft: MLXSTFT

  init(weights: [String: MLXArray],
       styleDim: Int,
       resblockKernelSizes: [Int],
       upsampleRates: [Int],
       upsampleInitialChannel: Int,
       resblockDilationSizes: [[Int]],
       upsampleKernelSizes: [Int],
       genIstftNFft: Int,
       genIstftHopSize: Int)
  {
    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count

    let upsampleScaleNum = MLX.product(MLXArray(upsampleRates)) * genIstftHopSize
    let upsampleScaleNumVal: Int = upsampleScaleNum.item()

    mSource = SourceModuleHnNSF(
      weights: weights,
      samplingRate: KokoroTTS.Constants.samplingRate,
      upsampleScale: upsampleScaleNum.item(),
      harmonicNum: 8,
      voicedThreshold: 10
    )

    f0Upsample = Upsample(scaleFactor: .float(Float(upsampleScaleNumVal)))

    noiseConvs = []
    noiseRes = []
    ups = []

    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      ups.append(
        ConvWeighted(
          weightG: weights["decoder.generator.ups.\(i).weight_g"]!,
          weightV: weights["decoder.generator.ups.\(i).weight_v"]!,
          bias: weights["decoder.generator.ups.\(i).bias"]!,
          stride: u,
          padding: (k - u) / 2
        )
      )
    }

    resBlocks = []
    for i in 0 ..< ups.count {
      let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
      for (j, (k, d)) in zip(resblockKernelSizes, resblockDilationSizes).enumerated() {
        resBlocks.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.resblocks.\((i * resblockKernelSizes.count) + j)",
            channels: ch,
            kernelSize: k,
            dilation: d,
            styleDim: styleDim
          )
        )
      }

      let cCur = ch
      if i + 1 < upsampleRates.count {
        let strideF0: Int = MLX.product(MLXArray(upsampleRates)[(i + 1)...]).item()
        noiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: strideF0 * 2,
            stride: strideF0,
            padding: (strideF0 + 1) / 2,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )

        noiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 7,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      } else {
        noiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: 1,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )
        noiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 11,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      }
    }

    postNFFt = genIstftNFft

    convPost = ConvWeighted(
      weightG: weights["decoder.generator.conv_post.weight_g"]!,
      weightV: weights["decoder.generator.conv_post.weight_v"]!,
      bias: weights["decoder.generator.conv_post.bias"]!,
      stride: 1,
      padding: 3
    )

    reflectionPad = ReflectionPad1d(padding: (1, 0))

    stft = MLXSTFT(
      filterLength: genIstftNFft,
      hopLength: genIstftHopSize,
      winLength: genIstftNFft
    )
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ F0Curve: MLXArray) -> MLXArray {
    var f0New = F0Curve[.newAxis, 0..., 0...].transposed(0, 2, 1)
    f0New = f0Upsample(f0New)

    var (harSource, _, _) = mSource(f0New)

    harSource = MLX.squeezed(harSource.transposed(0, 2, 1), axis: 1)
    // BUILD 49: generator-internal telemetry — the excitation, before any
    // spectral shaping. Device audio is periodic-but-unshaped, so the first
    // question is whether the source itself is sane.
    KokoroDiagFlags.genStat("gen_harsource", harSource)
    let (harSpec, harPhase) = stft.transform(inputData: harSource)
    // BUILD 44→45 (E2, production default): materialise the source STFT
    // before the upsample chain consumes it. (The 45 G-arm CPU pin here was
    // falsified for the residual and removed in 46.)
    if KokoroDiagFlags.decoderBarrier { MLX.eval(harSpec, harPhase) }
    KokoroDiagFlags.genStat("gen_harspec", harSpec)
    KokoroDiagFlags.genStat("gen_harphase", harPhase)

    var har = MLX.concatenated([harSpec, harPhase], axis: 1)
    har = MLX.swappedAxes(har, 2, 1)
        
    var newX = x
    for i in 0 ..< numUpsamples {
      newX = LeakyReLU(negativeSlope: 0.1)(newX)
      // BUILD 54 CANDIDATE (cpuNoiseConvs): the noise branch is the largest
      // remaining device divergence after the predictor pin — xsrc0 measured
      // 1.95-2.21x Mac on every row. It runs at STFT-frame rate, not sample
      // rate, so it is one of the cheap pieces of the generator.
      var xSource: MLXArray
      if KokoroDiagFlags.cpuNoiseConvs {
        let harIn = har
        xSource = Device.withDefaultDevice(Device(.cpu)) {
          var t = noiseConvs[i](harIn)
          t = MLX.swappedAxes(t, 2, 1)
          t = noiseRes[i](t, s)
          MLX.eval(t)
          return t
        }
      } else {
        xSource = noiseConvs[i](har)
        xSource = MLX.swappedAxes(xSource, 2, 1)
        xSource = noiseRes[i](xSource, s)
      }
      KokoroDiagFlags.genStat("gen_xsrc\(i)", xSource)

      newX = MLX.swappedAxes(newX, 2, 1)
      newX = ups[i](newX, conv: MLX.convTransposed1d)
      newX = MLX.swappedAxes(newX, 2, 1)

      if i == numUpsamples - 1 {
        newX = reflectionPad(newX)
      }
      newX = newX + xSource
      
      var xs: MLXArray?
      for j in 0 ..< numKernels {
        if xs == nil {
          xs = resBlocks[i * numKernels + j](newX, s)
        } else {
          let temp = resBlocks[i * numKernels + j](newX, s)
          xs = xs! + temp
        }
      }
      newX = xs! / numKernels
      // BUILD 49: per-upsample-group output — the shaping chain, block by
      // block. Where device/Mac ratios diverge names the failing stage.
      KokoroDiagFlags.genStat("gen_up\(i)", newX)
      // BUILD 44 (E2): materialise after each upsample block group.
      if KokoroDiagFlags.decoderBarrier { MLX.eval(newX) }
    }

    newX = LeakyReLU(negativeSlope: 0.01)(newX)

    newX = MLX.swappedAxes(newX, 2, 1)
    // BUILD 50 — THE FIX. Build-49 telemetry localised the device fault to
    // exactly this convolution. Every stage before it matches Mac within
    // 0.85-1.2x; convPost's output collapses to 0.12-0.16x of Mac RMS on all
    // 8 rows, and its DYNAMIC RANGE is destroyed: Mac [-36.6, +14.4], device
    // [-4.1, +5.0]. Since spec = exp(firstHalf), Mac's large negatives
    // exponentiate to true spectral valleys (formants); the device's squashed
    // range floors the spectrum at ~0.016 — a near-flat magnitude envelope.
    // That is precisely the measured device audio: pitch intact, no formants,
    // 76-91% of energy above 6 kHz. One wrong Metal conv variant on A-series
    // (mlx #2205 class), one utterly broken voice.
    // Fix: run this single small conv on the CPU stream. Unified memory makes
    // the transfer free; it is one layer on a [1, 22, frames] tensor.
    let preConv = newX
    if KokoroDiagFlags.cpuPostConv {
      newX = Device.withDefaultDevice(Device(.cpu)) {
        let r = convPost(preConv, conv: MLX.conv1d)
        MLX.eval(r)
        return r
      }
    } else {
      newX = convPost(preConv, conv: MLX.conv1d)
    }
    newX = MLX.swappedAxes(newX, 2, 1)
    KokoroDiagFlags.genStat("gen_postconv", newX)
    
    let spec = MLX.exp(newX[0..., 0 ..< (postNFFt / 2 + 1), 0...])
    let phase = MLX.sin(newX[0..., (postNFFt / 2 + 1)..., 0...])
    // BUILD 44→45 (E2, production default): materialise around the inverse
    // STFT — the last internal boundary before samples exist.
    if KokoroDiagFlags.decoderBarrier { MLX.eval(spec, phase) }
    KokoroDiagFlags.genStat("gen_spec", spec)
    KokoroDiagFlags.genStat("gen_phase", phase)

    let result = stft.inverse(magnitude: spec, phase: phase)
    if KokoroDiagFlags.decoderBarrier { MLX.eval(result) }
    KokoroDiagFlags.genStat("gen_result", result)
    return result
  }
}
