//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXUtilsLibrary

/// Main class that encapsulates the complete Kokoro text-to-speech pipeline.
///
/// KokoroTTS converts text input into audio output by:
/// 1. Processing text through grapheme-to-phoneme (G2P) conversion
/// 2. Encoding the phonemes using BERT-based embeddings
/// 3. Predicting duration and prosody for natural speech
/// 4. Generating audio through a decoder network
///
/// Example usage:
/// ```swift
/// let tts = KokoroTTS(modelPath: modelURL, g2p: .misaki)
/// let audioData = try tts.generateAudio(voice: voiceEmbedding,
///                                       language: .english,
///                                       text: "Hello world",
///                                       speed: 1.0)
/// ```
public final class KokoroTTS {
  /// Errors from the TTS side
  public enum KokoroTTSError: Error {
    /// Thrown when input text exceeds maximum token count
    case tooManyTokens
  }

  // BUILD 43→44 (2026-08-25) — device-corruption hunt: instrumentation + probes.
  // Build-43 run-3 established: corruption enters at the decoder/vocoder (only
  // the audio stage is pass-unstable), stochastic, sustained-load sensitive,
  // spikes biased to the buffer head; the token stage additionally diverges
  // device-vs-Mac (G2P via OS NLTagger + a gated numeric suspect). Build-44
  // levers: token-identity diag (always on), Z safe-framing and E2 decoder
  // barriers (KokoroDiagFlags), F frame-stage padding (below). Token-stage
  // bucket padding (build-43 "P") is REMOVED: it crashed on device (SIGSEGV)
  // and measurably compressed real-token durations 10-32% through the biLSTM
  // backward pass — falsified twice over.

  /// Per-synthesis diagnostics, populated on every generateAudio call.
  public struct SynthDiag {
    /// Token count incl. the two boundary zeros.
    public var realTokens = 0
    /// FNV-1a-64 of the token IDs (hex) — G2P identity across platforms.
    public var tokensHash = ""
    /// First 16 token IDs — G2P divergence localisation.
    public var tokensHead: [Int] = []
    /// Decoder frames from predicted durations (pre frame-padding).
    public var totalFrames = 0
    /// Frame-sequence length the decoder actually ran (== totalFrames unless
    /// frameStagePad).
    public var paddedFrames = 0
    /// Stage name -> [min, max, rms]. Only when collectStageStats is on.
    public var stageStats: [String: [Float]] = [:]
    public init() {}
  }

  /// PROBE F — pad the FRAME-stage tensors (asr / F0 / N) to the next bucket
  /// in {128, 256, 384, 512} with zeros AFTER alignment, then trim the audio
  /// back at the true-frame boundary. Token stage untouched — the biLSTM
  /// contamination that falsified token padding cannot occur here; durations
  /// are computed before padding and are unchanged.
  public var frameStagePad = false

  /// PROBE — force materialisation between pipeline stages. If corruption
  /// vanishes under barriers, the defect is a lazy-graph/fusion race, not a
  /// kernel-variant miscompute.
  public var evalBarrier = false

  /// PROBE — collect per-stage min/max/rms into lastDiag.stageStats for
  /// golden comparison against a Mac reference run. NOTE: computing stats
  /// forces evaluation (item() syncs), so stats themselves act as a partial
  /// barrier — run with stats OFF to observe the undisturbed pipeline.
  public var collectStageStats = false

  /// Diagnostics of the most recent generateAudio call.
  public private(set) var lastDiag = SynthDiag()

  /// BERT model for encoding phoneme sequences
  private let bert: CustomAlbert!
  
  /// Linear layer to project BERT embeddings
  private let bertEncoder: Linear!
  
  /// Encoder for duration prediction features
  private let durationEncoder: DurationEncoder!
  
  /// Bidirectional LSTM for duration prediction
  private let predictorLSTM: LSTM!
  
  /// Projection layer for final duration values
  private let durationProj: Linear!
  
  /// Predictor for prosodic features (F0, pitch)
  private let prosodyPredictor: ProsodyPredictor!
  
  /// Text encoder that processes phoneme sequences
  private let textEncoder: TextEncoder!
  
  /// Decoder that generates audio from encoded features
  private let decoder: Decoder!
  
  /// Grapheme-to-phoneme processor for text conversion
  private let g2pProcessor: G2PProcessor?
  
  /// Currently active language (cached to avoid reinitializing G2P)
  private var chosenLanguage: Language = .none
  
  /// Initializes the Kokoro TTS engine with model weights and G2P processor.
  /// - Parameters:
  ///   - modelPath: URL to the directory containing model weights
  ///   - g2p: Grapheme-to-phoneme processor type (default: Misaki)
  public init(modelPath: URL, g2p: G2P = .misaki) {
    // Load and sanitize model weights
    let sanitizedWeights = WeightLoader.loadWeights(modelPath: modelPath)
    let config = KokoroConfig.loadConfig()
    
    // Initialize BERT model for phoneme encoding
    bert = CustomAlbert(
      weights: sanitizedWeights,
      config: AlbertModelArgs(
        numHiddenLayers: config.plbert.numHiddenLayers,
        numAttentionHeads: config.plbert.numAttentionHeads,
        hiddenSize: config.plbert.hiddenSize,
        intermediateSize: config.plbert.intermediateSize,
        vocabSize: config.nToken
      )
    )
    
    // Initialize BERT output encoder
    bertEncoder = Linear(
      weight: sanitizedWeights["bert_encoder.weight"]!,
      bias: sanitizedWeights["bert_encoder.bias"]!
    )
    
    // Initialize duration prediction components
    durationEncoder = DurationEncoder(
      weights: sanitizedWeights,
      dModel: config.hiddenDim,
      styDim: config.styleDim,
      nlayers: config.nLayer
    )

    // Initialize bidirectional LSTM for duration prediction
    predictorLSTM = LSTM(
      inputSize: config.hiddenDim + config.styleDim,
      hiddenSize: config.hiddenDim / 2,
      wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
      whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
      biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
      biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
      wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
      whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
    )

    // Initialize duration projection layer
    durationProj = Linear(
      weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
      bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
    )

    // Initialize prosody predictor (F0, pitch, etc.)
    prosodyPredictor = ProsodyPredictor(
      weights: sanitizedWeights,
      styleDim: config.styleDim,
      dHid: config.hiddenDim
    )

    // Initialize text encoder
    textEncoder = TextEncoder(
      weights: sanitizedWeights,
      channels: config.hiddenDim,
      kernelSize: config.textEncoderKernelSize,
      depth: config.nLayer,
      nSymbols: config.nToken
    )

    // Initialize audio decoder
    decoder = Decoder(
      weights: sanitizedWeights,
      dimIn: config.hiddenDim,
      styleDim: config.styleDim,
      dimOut: config.nMels,
      resblockKernelSizes: config.istftNet.resblockKernelSizes,
      upsampleRates: config.istftNet.upsampleRates,
      upsampleInitialChannel: config.istftNet.upsampleInitialChannel,
      resblockDilationSizes: config.istftNet.resblockDilationSizes,
      upsampleKernelSizes: config.istftNet.upsampleKernelSizes,
      genIstftNFft: config.istftNet.genIstftNFFT,
      genIstftHopSize: config.istftNet.genIstftHopSize
    )

    // Initialize G2P processor for text-to-phoneme conversion
    g2pProcessor = try? G2PFactory.createG2PProcessor(engine: g2p)
  }
  
  /// Generates audio from text using the specified voice and parameters.
  ///
  /// This method performs the complete TTS pipeline:
  /// 1. Converts text to phonemes (G2P)
  /// 2. Tokenizes and encodes phonemes
  /// 3. Predicts duration and prosody
  /// 4. Generates audio waveform
  ///
  /// - Parameters:
  ///   - voice: Voice embedding array (contains speaker characteristics)
  ///   - language: Target language for pronunciation
  ///   - text: Input text to synthesize
  ///   - speed: Speech speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
  /// - Returns: Array of audio samples as Float values
  /// - Throws: `KokoroTTSError.tooManyTokens` if text is too long,
  ///           or `G2PProcessorError` if G2P processing fails
  public func generateAudio(voice: MLXArray, language: Language, text: String, speed: Float = 1.0) throws -> ([Float], [MToken]?) {
    // Update language if it has changed
    try updateLanguageIfNeeded(language)

    // Start performance timing
    BenchmarkTimer.reset()
    BenchmarkTimer.startTimer(Constants.bm_TTS)

    // Step 1: Convert text to phonemes
    let (phonemizedText, tokenArray) = try phonemizeText(text)

    // Step 2: Tokenize and prepare input
    let (paddedInputIds, attentionMask, inputLengths, textMask, inputIds, realCount) = try prepareInputTensors(phonemizedText)

    var diag = SynthDiag()
    diag.realTokens = realCount
    // Build 44 — token identity (always on, host-side, cheap): the G2P stacks
    // proved OS-divergent (NLTagger); equal token COUNTS do not imply equal
    // token IDs, so cross-platform comparisons key on this hash.
    var h: UInt64 = 0xcbf29ce484222325
    for t in [0] + inputIds + [0] {
      h = (h ^ UInt64(truncatingIfNeeded: t)) &* 0x100000001b3
    }
    diag.tokensHash = String(format: "%016llx", h)
    diag.tokensHead = Array(inputIds.prefix(16))

    // Step 3: Extract style embeddings from voice
    let (globalStyle, acousticStyle) = extractStyleEmbeddings(from: voice, tokenCount: inputIds.count)

    // Steps 4+5: BERT → duration features → durations (+ alignment).
    // BUILD 45 DIAGNOSTIC (cpuDurationHead): whole token→durations path on
    // the CPU stream. NOT a fix — ground truth (upstream Kokoro PyTorch)
    // matches MAC-GPU durations sample-exactly; device-GPU AND CPU both
    // diverge (dan: 116 truth vs 96 device-GPU vs 55 CPU). Retained so the
    // bench can measure whether CPU-durations move the corruption. 46 pacing
    // fix = deterministic plain-Swift duration head. eval() INSIDE the scope:
    // mlx ops capture the default stream at construction.
    func durationPath() -> (MLXArray, MLXArray, MLXArray) {
      let df = encodeBERTAndDuration(
        inputIds: paddedInputIds,
        attentionMask: attentionMask,
        inputLengths: inputLengths,
        textMask: textMask,
        style: globalStyle
      )
      let (pd, at) = predictDurations(
        features: df, batchSize: paddedInputIds.shape[1], speed: speed)
      return (df, pd, at)
    }
    let durationFeatures: MLXArray
    let predictedDurations: MLXArray
    let alignmentTarget: MLXArray
    if KokoroDiagFlags.cpuDurationHead {
      (durationFeatures, predictedDurations, alignmentTarget) =
        Device.withDefaultDevice(Device(.cpu)) {
          let r = durationPath()
          eval(r.0, r.1, r.2)
          return r
        }
    } else {
      (durationFeatures, predictedDurations, alignmentTarget) = durationPath()
    }
    barrier(durationFeatures)
    stat("dur_features", durationFeatures, into: &diag)
    barrier(predictedDurations, alignmentTarget)
    stat("durations", predictedDurations.asType(.float32), into: &diag)

    // Step 6: Generate aligned encodings
    let alignedEncoding = durationFeatures.transposed(0, 2, 1).matmul(alignmentTarget)
    barrier(alignedEncoding)
    stat("aligned", alignedEncoding, into: &diag)

    // Step 7: Predict prosody (F0, pitch)
    let (f0Prediction, nPrediction) = prosodyPredictor.F0NTrain(x: alignedEncoding, s: globalStyle)
    barrier(f0Prediction, nPrediction)
    stat("f0", f0Prediction, into: &diag)
    stat("n", nPrediction, into: &diag)

    // Step 8: Encode text for decoder
    let textEncoding = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    barrier(textEncoding)
    stat("text_enc", textEncoding, into: &diag)
    var asrFeatures = MLX.matmul(textEncoding, alignmentTarget)
    barrier(asrFeatures)
    stat("asr", asrFeatures, into: &diag)

    // Build 44 — PROBE F: frame-stage padding. Zero-pad asr / F0 / N along
    // the frame axis to the next bucket AFTER alignment. Durations were
    // computed above and are untouched; the biLSTM never sees the pads.
    // F0/N run at a multiple of the frame rate — pad proportionally.
    let totalFrames = alignmentTarget.dim(-1)
    diag.totalFrames = totalFrames
    var paddedFrames = totalFrames
    var f0In = f0Prediction
    var nIn = nPrediction
    if frameStagePad {
      let buckets = [128, 256, 384, 512]
      if let bucket = buckets.first(where: { $0 >= totalFrames }), bucket > totalFrames, totalFrames > 0 {
        let padF = bucket - totalFrames
        asrFeatures = MLX.padded(
          asrFeatures, widths: [IntOrPair([0, 0]), IntOrPair([0, 0]), IntOrPair([0, padF])])
        let f0Ratio = f0Prediction.dim(-1) / totalFrames
        let nRatio = nPrediction.dim(-1) / totalFrames
        f0In = MLX.padded(
          f0Prediction, widths: [IntOrPair([0, 0]), IntOrPair([0, padF * max(1, f0Ratio)])])
        nIn = MLX.padded(
          nPrediction, widths: [IntOrPair([0, 0]), IntOrPair([0, padF * max(1, nRatio)])])
        paddedFrames = bucket
      }
    }
    diag.paddedFrames = paddedFrames

    // Step 9: Generate audio
    let audio = decoder(
      asr: asrFeatures,
      F0Curve: f0In,
      N: nIn,
      s: acousticStyle
    )[0]
    stat("audio", audio, into: &diag)

    // Try to predict timestamp of each token if G2P processor returns tokens
    if let tokenArray {
      TimestampPredictor.preditTimestamps(tokens: tokenArray, predictionDuration: predictedDurations)
    }

    var samples = audio[0].asArray(Float.self)

    // Build 44 — trim the frame-padding tail at the true-frame boundary.
    // Guards make the slice provably in-bounds; divisibility failure degrades
    // to no-trim (audible pad tail), never a crash.
    if paddedFrames > totalFrames, totalFrames > 0, samples.count % paddedFrames == 0 {
      let samplesPerFrame = samples.count / paddedFrames
      let keep = min(samplesPerFrame * totalFrames, samples.count)
      samples = Array(samples[0 ..< keep])
    }
    lastDiag = diag

    // Stop performance timing
    BenchmarkTimer.stopTimer(Constants.bm_TTS)

    return (samples, tokenArray)
  }

  /// Build 43 — force materialisation between stages when evalBarrier is on.
  private func barrier(_ arrays: MLXArray...) {
    guard evalBarrier else { return }
    for a in arrays { eval(a) }
  }

  /// Build 43 — record [min, max, rms] for a stage when collectStageStats is on.
  private func stat(_ name: String, _ x: MLXArray, into diag: inout SynthDiag) {
    guard collectStageStats else { return }
    let f = x.asType(.float32)
    let mn: Float = f.min().item()
    let mx: Float = f.max().item()
    let rms: Float = MLX.sqrt(MLX.mean(f * f)).item()
    diag.stageStats[name] = [mn, mx, rms]
  }
  
  /// Updates the G2P language if it differs from the current language.
  private func updateLanguageIfNeeded(_ language: Language) throws {
    guard chosenLanguage != language else { return }
    
    guard let g2pProcessor else {
      throw G2PProcessorError.processorNotInitialized
    }
    
    try g2pProcessor.setLanguage(language)
    chosenLanguage = language
  }
  
  /// Converts input text to phonemes using the G2P processor.
  private func phonemizeText(_ text: String) throws -> (String, [MToken]?) {
    let phonemizedOutput = try g2pProcessor?.process(input: text)
    guard let phonemizedOutput else {
      throw G2PProcessorError.processorNotInitialized
    }
    return phonemizedOutput
  }
  
  /// Prepares input tensors for the model from phonemized text.
  /// - Returns: Tuple containing:
  ///   - paddedInputIds: Tokenized input sequence with boundary zeros
  ///   - attentionMask: Mask for attention mechanism
  ///   - inputLengths: Length of input sequence
  ///   - textMask: Mask for text padding
  ///   - inputIds: Original token IDs before boundary zeros
  ///   - realCount: token count incl. boundary zeros
  /// Build 44: token-stage bucket padding (build-43 "P") REMOVED — device
  /// SIGSEGV + biLSTM duration contamination. Pre-43 tensor semantics hold:
  /// sequence length == realCount, masks flag nothing.
  private func prepareInputTensors(_ phonemizedText: String) throws -> (MLXArray, MLXArray, MLXArray, MLXArray, [Int], Int) {
    // Tokenize phonemized text
    let inputIds = Tokenizer.tokenize(phonemizedText: phonemizedText)

    // Check token count limit
    guard inputIds.count <= Constants.maxTokenCount else {
      throw KokoroTTSError.tooManyTokens
    }

    // Add padding tokens at start and end
    let paddedInputIdsArray = [0] + inputIds + [0]
    let realCount = paddedInputIdsArray.count

    let paddedInputIds = MLXArray(paddedInputIdsArray).expandedDimensions(axes: [0])

    // Create input length tensor
    let inputLengths = MLXArray(realCount)
    let inputLengthMax: Int = paddedInputIds.dim(-1)

    // Create text mask for padding positions
    var textMask = MLXArray(0 ..< inputLengthMax)
    textMask = textMask + 1 .> inputLengths
    textMask = textMask.expandedDimensions(axes: [0])

    // Create attention mask (1 for valid positions, 0 for padding)
    let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
    let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
    let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)

    return (paddedInputIds, attentionMask, inputLengths, textMask, inputIds, realCount)
  }
  
  /// Extracts style embeddings from the voice array.
  /// - Parameters:
  ///   - voice: Voice embedding array
  ///   - tokenCount: Number of tokens in the input
  /// - Returns: Tuple of (globalStyle, acousticStyle)
  ///   - globalStyle: Style embedding for prosody/duration (indices 128+)
  ///   - acousticStyle: Style embedding for acoustic features (indices 0-127)
  private func extractStyleEmbeddings(from voice: MLXArray, tokenCount: Int) -> (MLXArray, MLXArray) {
    // Extract reference style from voice embedding
    let referenceStyle = voice[tokenCount - 1, 0 ... 1, 0...]
    
    // Split into global style (for prosody/duration) and acoustic style
    let globalStyle = referenceStyle[0 ... 1, 128...]
    let acousticStyle = referenceStyle[0 ... 1, 0 ... 127]
    
    return (globalStyle, acousticStyle)
  }
  
  /// Encodes text with BERT and generates duration prediction features.
  private func encodeBERTAndDuration(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    inputLengths: MLXArray,
    textMask: MLXArray,
    style: MLXArray
  ) -> MLXArray {
    // Pass through BERT model
    let (bertOutput, _) = bert(inputIds, attentionMask: attentionMask)
    
    // Project BERT output and transpose for duration encoder
    let bertEncoded = bertEncoder(bertOutput).transposed(0, 2, 1)
    
    // Generate duration features with style conditioning
    let durationFeatures = durationEncoder(
      bertEncoded,
      style: style,
      textLengths: inputLengths,
      m: textMask
    )
    
    return durationFeatures
  }
  
  /// Predicts phoneme durations and creates alignment target matrix.
  /// - Parameters:
  ///   - features: Duration prediction features from encoder
  ///   - batchSize: Size of the input batch
  ///   - speed: Speech speed multiplier
  /// - Returns: Predicted durations and alignment target matrix for duration expansion
  private func predictDurations(features: MLXArray, batchSize: Int, speed: Float) -> (MLXArray, MLXArray) {
    // Pass through LSTM
    let (lstmOutput, _) = predictorLSTM(features)
    
    // Project to duration values
    let durationLogits = durationProj(lstmOutput)
    
    // Convert to actual durations (clamped to minimum of 1 frame)
    let durationSigmoid = MLX.sigmoid(durationLogits).sum(axis: -1) / speed
    let predictedDurations = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
    
    // Create alignment matrix
    return (predictedDurations, createAlignmentTarget(durations: predictedDurations, batchSize: batchSize))
  }
  
  /// Creates an alignment target matrix from predicted durations. Maps each phoneme to multiple frames based on duration.
  /// Each row corresponds to a phoneme, and columns represent frames.
  /// - Parameters:
  ///   - durations: Predicted duration for each phoneme
  ///   - batchSize: Size of the input batch
  /// - Returns: Alignment matrix [batchSize × totalFrames]
  private func createAlignmentTarget(durations: MLXArray, batchSize: Int) -> MLXArray {
    // Create indices array by repeating each index according to its duration
    let indices = MLX.concatenated(
      durations.enumerated().map { index, duration in
        let frameCount: Int = duration.item()
        return MLX.repeated(MLXArray([index]), count: frameCount)
      }
    )

    // Create one-hot encoded alignment matrix
    let totalFrames = indices.shape[0]
    var alignmentArray = [Float](repeating: 0.0, count: totalFrames * batchSize)
    
    for frame in 0 ..< totalFrames {
      let phonemeIndex: Int = indices[frame].item()
      alignmentArray[phonemeIndex * totalFrames + frame] = 1.0
    }
    
    let alignmentTarget = MLXArray(alignmentArray).reshaped([batchSize, totalFrames])
    return alignmentTarget.expandedDimensions(axis: 0)
  }
  
  /// Constants used throughout the TTS engine.
  public struct Constants {
    /// Maximum number of tokens allowed in input
    public static let maxTokenCount = 510
    
    /// Audio sampling rate in Hz
    public static let samplingRate = 24000
    
    // Benchmark timer identifiers
    public static let bm_TTS = "TTSAudio"
    static let bm_Phonemize = "Phonemize"
    static let bm_bert = "BERT"
    static let bm_duration = "Duration"
    static let bm_prosody = "Prosody"
    static let bm_decoder = "Decoder"
  }
}
