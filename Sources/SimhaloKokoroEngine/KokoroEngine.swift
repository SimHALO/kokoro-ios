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

// @unchecked Sendable: every access to the mutable state below goes through
// `lock` — Swift 6 cannot see that, so we assert it. A `static let shared`
// is immutable global state and passes strict concurrency.
public final class KokoroEngine: @unchecked Sendable {
  public static let shared = KokoroEngine()
  private init() {}
  private var tts: KokoroTTS?
  private var voices: [String: MLXArray] = [:]
  private let lock = NSLock()

  // A/B LEVER for the device-corruption hunt (build 42, 2026-08-25). The same
  // engine revision produces clean samples on Mac CPU AND Mac GPU (8/8 rows,
  // zero bad) while the phone bench shows ±50-69 spikes — device-side only.
  // The one cache behaviour unique to this facade is the 64MB cacheLimit +
  // clearCache-per-chunk below; a plausible interaction with mlx's lazy evals
  // / JIT kernel cache on A-series. `true` = discipline ON (current, build-34
  // jetsam fix). `false` = 1GB limit, no per-chunk purge — bench-only setting
  // to test whether the corruption follows the cache discipline. Expect a
  // higher jetsam risk while off; that is the experiment, not a regression.
  private var cacheDisciplineOn = true
  public var cacheDiscipline: Bool {
    get { lock.lock(); defer { lock.unlock() }; return cacheDisciplineOn }
    set {
      lock.lock(); defer { lock.unlock() }
      cacheDisciplineOn = newValue
      Memory.cacheLimit = newValue ? 64 * 1024 * 1024 : 1024 * 1024 * 1024
    }
  }

  public var isLoaded: Bool {
    lock.lock(); defer { lock.unlock() }
    return tts != nil && !voices.isEmpty
  }

  public var availableVoices: [String] {
    lock.lock(); defer { lock.unlock() }
    return voices.keys.map { $0.replacingOccurrences(of: ".npy", with: "") }.sorted()
  }

  public func load(modelPath: URL, voicesPath: URL) throws {
    lock.lock(); defer { lock.unlock() }
    guard let v = NpyzReader.read(fileFromPath: voicesPath), !v.isEmpty else {
      throw NSError(domain: "KokoroEngine", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "voices.npz unreadable"])
    }
    voices = v
    // MEMORY DISCIPLINE (Luke's build-34 bench: engine loaded and synthesized,
    // then iOS's watchdog silently killed the app mid-suite — no crash dialog,
    // the jetsam signature). MLX caches GPU buffers per generation and never
    // trims; on top of 312MB fp32 weights, back-to-back syntheses walk the
    // footprint over the limit. Cap the cache small — synthesis prefers
    // re-allocation over a corpse — and purge after every chunk below.
    // Build 42: honours the cacheDiscipline lever if it was set before load.
    Memory.cacheLimit = cacheDisciplineOn ? 64 * 1024 * 1024 : 1024 * 1024 * 1024
    tts = try withError { KokoroTTS(modelPath: modelPath, g2p: .misaki) }
  }

  public func synthesize(text: String, voiceId: String) throws -> (samples: [Float], sampleRate: Int) {
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
      // withError: MLX's internal errors (Metal allocation refusals included)
      // become THROWN Swift errors instead of the library's default death — on
      // device an MLX error otherwise surfaces as an anonymous fatalError
      // ("no resultOut pointer" / _assertionFailure, SIGTRAP), which is
      // precisely the crash Luke's build-36 bench produced while the identical
      // suite ran clean on a memory-rich Mac at RTF 14-19x. A thrown error
      // reaches the bridge, becomes a promise rejection, and the bench prints
      // the actual Metal message instead of dying.
      try autoreleasepool {
        let (audio, _) = try withError { try tts.generateAudio(voice: voice, language: language, text: chunk) }
        samples.append(contentsOf: audio)
      }
      // Return cached GPU buffers to the OS between chunks — the difference
      // between a bounded sawtooth and a monotonic climb into the watchdog.
      // Build 42: skipped when the cacheDiscipline lever is OFF (A/B only).
      if cacheDisciplineOn { Memory.clearCache() }
    }
    return (samples, KokoroTTS.Constants.samplingRate)
  }

  func sentenceChunks(_ text: String) -> [String] {
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
    if parts.isEmpty { parts = [trimmed] }
    // SUB-SENTENCE SPLIT for long single sentences. Luke's device crash
    // (build 37, symbolicated): mlx::core::Full::eval_gpu failing mid-fill on
    // a 225-char one-sentence checklist line producing ~15s of audio in ONE
    // generation — the decoder's intermediates blow the phone's GPU allocator
    // while a Mac runs the identical line clean. Same shape as the lineage's
    // documented long-clip OOM, at the phone's threshold. Splitting at clause
    // boundaries caps each generation at a few seconds of audio; the pause at
    // a comma is where a human reader breathes anyway.
    var bounded: [String] = []
    for part in parts {
      if part.count <= 140 { bounded.append(part); continue }
      var piece = ""
      for ch in part {
        piece.append(ch)
        if piece.count >= 90, ch == "," || ch == ";" || ch == "\u{2014}" {
          bounded.append(piece.trimmingCharacters(in: .whitespaces))
          piece = ""
        }
      }
      let rest = piece.trimmingCharacters(in: .whitespaces)
      if !rest.isEmpty {
        if let last = bounded.last, rest.count < 40, last.count + rest.count < 160 {
          bounded[bounded.count - 1] = last + " " + rest
        } else {
          bounded.append(rest)
        }
      }
    }
    return bounded.isEmpty ? [trimmed] : bounded
  }
}
