class SpeechToTextCli < Formula
  desc "Local real-time speech-to-text for Apple Silicon"
  homepage "https://github.com/mexanichp/speech-to-text-cli"
  url "https://github.com/mexanichp/speech-to-text-cli/releases/download/v0.1.0/speech-to-text-cli-0.1.0-aarch64-apple-darwin.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  # Builds the pinned Python environment the two sidecars run in, on first use.
  # Not a set of `resource` stanzas: that would have Homebrew build numpy, scipy
  # and tokenizers from sdists, and every one of them ships an arm64 wheel.
  depends_on "uv"

  def install
    bin.install "speech-to-text-cli"
  end

  def caveats
    <<~EOS
      First run does two things once, and both need the network:

        - builds a Python environment under ~/.cache/speech-to-text-cli
        - downloads the two models, about 4 GB, to ~/.cache/huggingface

      Microphone access is granted to the terminal running this, not to the
      binary, so the prompt names Terminal or iTerm.
    EOS
  end

  test do
    assert_match "speech-to-text", shell_output("#{bin}/speech-to-text-cli --version")
  end
end
