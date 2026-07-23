- winit の request_redraw だけ ... redraw は 18219 回
- wgpu の no sync 指定 ... redraw は 706, render が返ってくるのがだいたい 1.2 ms
- wgpu の vsync 指定 ... redraw は 61, render が返ってくるのがだいたい 16 ms

vsync を使うと、 CPU 側は `get_current_texture()` のところで vsync のために GPU が描画し終わって次のフレームを渡すまで待つことになるため、
重い処理が loop 内に入ると 60 fps を超えられなくなる。
