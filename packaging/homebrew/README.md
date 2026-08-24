# homebrew

Publishing a release and updating the tap.

Create the tap once. The repository name must start with `homebrew-`.

```sh
gh repo create mexanichp/homebrew-tap --public
git clone git@github.com:mexanichp/homebrew-tap.git
mkdir -p homebrew-tap/Formula
cp packaging/homebrew/speech-to-text-cli.rb homebrew-tap/Formula/
```

Cut a release. The workflow tests, builds, uploads the tarball and prints the
`url`, `sha256` and `version` lines to paste into the formula.

```sh
cargo set-version 0.1.1   # or edit Cargo.toml
git tag v0.1.1 && git push --tags
```

Install.

```sh
brew install mexanichp/tap/speech-to-text-cli
```

Check the formula before pushing it.

```sh
brew audit --strict --online mexanichp/tap/speech-to-text-cli
brew install --build-from-source mexanichp/tap/speech-to-text-cli
brew test mexanichp/tap/speech-to-text-cli
```

| why not | reason |
|---|---|
| homebrew-core | Apple Silicon only, and it downloads models on first run |
| `resource` stanzas for Python | Homebrew builds them from sdists; every pin here has an arm64 wheel |
| a cask | this is a CLI binary, not an app bundle |
