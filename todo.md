1. add multi-thread support for multi-window rendering: not very possible since OpenGL is not well supported for multi-thread
2. add demo for one simulation, multiple view -- Done for Windows
3. add demo for rendering a texture -- Done
4. cuda render buffer impl: for ray tracing or sth else that renders to a gpu memory buffer -- Done
5. **BUG**: on ubuntu, callback functions may be called multiple times between two frames
6. **BUG**: GPU memory leak in demo RenderTexture -- Done