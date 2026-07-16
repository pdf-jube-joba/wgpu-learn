# Boids

`winit` で固定サイズのウィンドウを作り、boids の分離・整列・結合を
`wgpu` の compute shader で計算します。AoS 形式の2本のストレージバッファを
ping-pong し、計算結果をそのまま頂点バッファとして使って700羽を描画します。

```console
cargo run --example boids --release
```

終了するにはウィンドウを閉じます。
