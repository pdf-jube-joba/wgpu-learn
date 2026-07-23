winit による抽象化が入っているのでややわかりにくい。
wgpu 自体は、
- device の情報が必要
- surface の情報が必要

一方で winit 側からすると、以下の事情でコードが複雑になりがち。
- ウィンドウを初期化する
- イベントループを作る
- 必要に応じて再レンダリング

## winit と wgpu の債務について
vsync というのがあって、ディスプレイの描画に合わせて描画内容を決めることができる。
面白いのが、これを制御するのは winit ではなくて wgpu の方らしい。
wgpu による `present_mode` で vsync に対応する FiFo を設定すると、
この時点でディスプレイの描画に強制的に同期させられる。
そのため、 winit 自体にはフレームレートを制御する仕組みがなくても、
やばい buzy_loop みたいにはならない。
（もしこれがなかったら、
winit で RedrawRequested で呼ばれるたびに request_redraw をしているので、
OS が馬鹿なら休眠とかしないで無制限にループするので、 CPU 使用率がやばくなる？）

## wgpu について
### `new(Arc<Window>)` からの流れ
- `wgpu::Instance::default()` で、 "インスタンス" を得る。これは実質的には wgpu のランタイムのことで、本当の gpu ドライバーとの橋渡しをする役割っぽい。バックエンドを選んだりもここからやれる。生成された `device` やらにハンドルが入っているらしい。
- これに `winit::Window` を渡して `wgpu::Surface` を作る。
- adapter では得た surface との compatibility を調べる必要があるので、 `request_adapter` のために先に Surface を作っている。
## render の中身
- `get_current_texture()` はこれから書くための texture を返す。
- `device` から command 記述してコンパイルする用の encoder を得る
  - このコマンドは一回分のため、 `clone()` もできないし、 submit によって消費される。
- queue に submit すると、 GPU に送れたことが分かった時点で return する。 GPU が描画したときではない。
- `frame.present()` も単に提出を行っているので、 render を抜けた時点で GPU によるディスプレイへの描画が完成しているわけじゃない。
  - この結果、 vsync が指定されない場合には window のイベントループ自体はディスプレイに合わせずにともかく速く回れるだけ回る（GPU タスクを投げ続ける）が、その場合には、描画するべき frame 自体が `get_current_texture()` で得られなくなる。
## wgpu とバックエンド
- vulkan とか direct 3d とかがあるらしい。これをコンパイルされたプログラムが実行時に選ぶ。wgpu が頑張ってシステムに問い合わせるらしい。

## わからないこと
- `adapter` から `device` と `queue` をとる？ `device` から取らない理由は？
- `config` を `surface` と `adapter` からとる理由は？ `device` から取らない理由は？
- しかも、 config は後でわざわざ設定している...直接 `surface.config.hoge = ...` はできないのかな？

# webgpu との違い
- 例えば、 commandencoder から得られる pass について、 webgpu では `pass.end()` を書く必要があるが、 wgpu ではスコープを抜けると自動で行われる。このために、 `_render_pass` は特殊な Drop 実装が存在するということになるため、 lifetime checker は勝手に寿命を決めることができず、ブロックの最後まで生き続けると仮定する必要がある。なので、 `{` で囲わないと、コンパイルが通らなくなる。
