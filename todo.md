1. - [ ] add multi-thread support for multi-window rendering: not very possible since OpenGL is not well supported for multi-thread
2. - [ ] add demo for one simulation, multiple view
   1. - [x] on Windows
   2. - [ ] on Ubuntu
3. - [x] add demo for rendering a texture
4. - [x] cuda render buffer impl: for ray tracing or sth else that renders to a gpu memory buffer
5. - [ ] **BUG**: on ubuntu, callback functions may be called multiple times between two frames
6. - [ ] **BUG**: GPU memory leak in demo RenderTexture -- Done
7. - [x] adapt for non-cuda devices (except for RenderTexture since it does not need this tech)
8. - [ ] add multi-thread for multi-window (multi-OpenGL context) and multi-imgui context rendering system
   1. - [ ] the guis for the same gui should be rendered in the same thread as the OpenGL renderings
   2. - [ ] wait for offical imgui to support multi-context
9. - [x] FIX: imgui doesn't support callback for multiple window guis.
   1. - [x] a new window manager with imgui context
   2. - [x] cannot destroy window when there are multiple windows (imgui contexts)

