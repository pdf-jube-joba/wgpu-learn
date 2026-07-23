# wgpu-learn

`wgpu`を小さな独立crateで試しながら学習するCargo workspaceです。

| crate | 内容 | 実行例 |
| --- | --- | --- |
| `basic-window` | `winit`とsurfaceの基本 | `cargo run -p basic-window` |
| `boids` | compute shaderと描画の連携 | `cargo run -p boids --release` |
| `compute` | compute shaderによる熱拡散 | `cargo run -p compute --release` |
| `measure-fps` | `winit`、FIFO、同期なしの比較 | `cargo run -p measure-fps --bin wgpu_fifo --release` |
| `nn-and-raytracing` | GPUレイトレーシングと3層NNの学習 | `cargo run -p nn-and-raytracing --release` |

NN用の生成画像を確認する場合:

```console
cargo run -p nn-and-raytracing --bin preview --release
```

workspace全体の確認:

```console
cargo check --workspace
```
