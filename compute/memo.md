compute shader の使い方（ WGSL） の example として、熱拡散方程式を二次元でやってみる例。
- `[N * N; f32]` の配列を用意する
- `a[i][j] = a[i][j] + alpha * (a[i + 1][j] + a[i - 1][j] + a[i][j + 1] + a[i][j - 1] - 4 * a[i][j])` で更新していく。
- なお、 GPU を生かしやすいように、 buffer を swap する方法を使う。

# 内容について

## 初期化まで
- (device, queue) を作る
  - surface がないので、 adaptor を作るときに compatibility を考えなくてよい。
  - buffer を device と descriptor から作る。これはわかりやすい。
## buffer と pipeline を作る。
`wgpu::Buffer` は GPU バッファへのポインタ
ここには型とかはなくて、 GPU から見たらただの用途とサイズのあるバイト列。

- current を作る。
  - 今回は `current` の buffer を最初から初期化している： `mapped_at_creation`
  - `buffer.slice(..).get_mapped_range_mut()` は、これに書き込むたびに GPU に送られる **のではなくて** CPU 側で保管して、 `unmap()` のときに一気に書き込む。
  - `push_error_scope` では、
    1. あるスコープ（`scope` として得られた GurdedObject が drop されるまで）の範囲内で、
    2. 指定したエラーを
    3. wgpu の panic にせずに、 Result としてキャッチするために使う。
- next を作るのは、 `mapped_at_creation` なしでいい。 `params` は `current` と同じようにやる。 `readback` も GPU から持ってくる用の buffer になる。
- bind 周りはちょっと別でやる。
- compute pipe line を作る
  - shader は wgsl で書いている。この wgsl 内でも `group` とか `binding` があって、 bindgroupkayout はこれと compatible になる必要がある。
  - pipeline_layout は bindgrouplayout だけ実質的には必要になっている。
- bindgroup を作るとき。
  - `BindGroupLayout` を一緒に Descriptor に渡す。これと実際の entry の整合性は必要。
  - `create_bind_group()` という関数を新たに作っていて、これを使っているときに `current` と `next` を入れ替えているのが面白い。
## シミュレーション
- encoder に対しては次のことをする。
- pass を完成させる
  - pass を作る
  - pipeline を設定する。これにはすでに、コンパイル済みシェーダープログラム、 `BindGroupLayout` の情報がある。
  - `BindGroup` を設定する。これは Layout と compatible な必要がある。ただし、 compatibility については型の関係性のところの注意を読む。
  - `dispatch_workgroups` で並列起動する（ compute shader 専用）：後述
  - `drop` されて、 pass は終了する。
- encoder 経由で buffer から buffer へ。

以上のようにしてスタックされた命令列を `encoder.finish()` で変換する？あるいは、変換中のオブジェクトに渡す？webgpu としてはここら辺はあまり規定されていなくて、とりあえず `queue.subgmit` に渡せるらしい。ただ、 `queue.submit()` が GPU が終わるまで待つとかはないのは以前にも書いた。
ここでは、 `wgpu::PollType::wait_indefinitely` して、 GPU の合流を待つ。

## 型の関係性
基本的には、全ては `FooDescriptor` から `Foo` を作る： `BufferDescriptor` から `Buffer`, `BindGroupLayoutDescriptor` から `BindGroupLayout`

以降では、 `FooDescriptor` と `Foo` は混ぜる。

| 作るもの | 必要なもの |
| --- | --- |
| `BindGroupLayout` | `[enum BindingType::Buffer]` の array |
| `BindGroup` | `BindingResource` <- `Buffer` の array |

> [!Warning] Layout は GPU オブジェクトのため、値オブジェクトっぽく中身の equality で比較できるわけではないらしい。
> 同じ Descriptor から作っても違う Layout オブジェクトになりうるので注意する。

## workgroup と invocation
workgroups も invocation も並列な実行を行う。
`pass.dispatch_workgroups()` は CPU 側から並列に呼び出す個数を書いていて、
WGSL の `@workgroups_size()` は1つの workgroup の中でいくつの invocation を出すかを記述している。

こうして workgroup * (invocation / workgroup) の数だけの invocation が行われて、これらの全ての invocation に対する unique な id が `@buildin(global_invocation_id)` になる。

| wgpu | CUDA |
| --- | --- |
| invocation | thread |
| workgroup | threadblock |
らしい。
また、同じ workgroup 内の invocation では、
`var<workgroup>` を共有できて `workgroupBarrier` で同期できて、 GPU 上で強調して処理する単位になるらしい。
