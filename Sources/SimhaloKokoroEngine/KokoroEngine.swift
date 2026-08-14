// MLX-FREE FACADE over KokoroSwift, and the ONLY product the SimHALO app links.
//
// Every type crossing this API is Foundation — no MLXArray reaches the app
// target. That single-consumer shape is what lets Xcode build the entire
// dependency graph (KokoroSwift, MLX, Cmlx, MisakiSwift, MLXUtils, and their
// tails) as STATIC archives folded into the app binary: four TestFlight builds
// established that separate dynamic images in this stack either go missing at
// launch (build 30) or poison React Native's exception symbolication into
// segfaults (builds 31-33). No images, no class of failure.

import Foundation
import MLX
import KokoroSwift
import MLXUtilsLibrary

public enum KokoroEngine {
  private static var tts: KokoroTTS?
  private static var voices: [String: MLXArray] = [:]
  private static let lock = NSLock()

  public static var isLoaded: Bool {
    lock.lock(); defer { lock.unlock() }
    return tts != nil && !voices.isEmpty
  }

  public static var availableVoices: [String] {
    lock.lock(); defer { lock.unlock() }
    return voices.keys.map { $0.replacingOccurrences(of: ".npy", with: "") }.sorted()
  }

  public static func load(modelPath: URL, voicesPath: URL) throws {
    lock.lock(); defer { lock.unlock() }
    guard let v = NpyzReader.read(fileFromPath: voicesPath), !v.isEmpty else {
      throw NSError(domain: "KokoroEngine", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "voices.npz unreadable"])
    }
    voices = v
    tts = KokoroTTS(modelPath: modelPath, g2p: .misaki)
  }

  public static func synthesize(text: String, voiceId: String) throws -> (samples: [Float], sampleRate: Int) {
    lock.lock(); defer { lock.unlock() }
    guard let tts = tts else {
      throw NSError(domain: "KokoroEngine", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "engine not loaded"])
    }
    guard let voice = voices["\(voiceId).npy"] ?? voices[voiceId] else {
      throw NSError(domain: "KokoroEngine", code: 5,
                    userInfo: [NSLocalizedDescriptionKey: "unknown voice \(voiceId)"])
    }
    let language: Language = voiceId.hasPrefix("b") ? .enGB : .enUS
    // Sentence chunking: the engine caps at 510 phonemes per call, and long
    // single generations OOM 4GB devices. Sim lines are 1-3 sentences.
    var samples: [Float] = []
    for chunk in sentenceChunks(text) {
      let (audio, _) = try tts.generateAudio(voice: voice, language: language, text: chunk)
      samples.append(contentsOf: audio)
    }
    return (samples, KokoroTTS.Constants.samplingRate)
  }

  static func sentenceChunks(_ text: String) -> [String] {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return [] }
    var parts: [String] = []
    var current = ""
    for ch in trimmed {
      current.append(ch)
      if ".!?".contains(ch) && current.count >= 40 {
        parts.append(current.trimmingCharacters(in: .whitespaces))
        current = ""
      }
    }
    let tail = current.trimmingCharacters(in: .whitespaces)
    if !tail.isEmpty {
      if let last = parts.last, tail.count < 40 {
        parts[parts.count - 1] = last + " " + tail
      } else {
        parts.append(tail)
      }
    }
    return parts.isEmpty ? [trimmed] : parts
  }
}
