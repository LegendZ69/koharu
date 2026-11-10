# koharu-desktop

`koharu-desktop` coordinates native page preparation for browser presentation
and export. `Desktop` owns the shared `koharu-renderer` and lazily the shared
native `koharu-rasterizer` used by export and previews. Each window has its own
`Presentation`, which retains that window's current frame, publishes frame
generations to the Tauri frontend, serves an exact-generation manifest and its
independently addressed resources, and validates durable transform commits.
A presentation is bound to one project and orders its page requests by project
revision: a request for older state is rejected, prefetches never supersede
presentation, and obsolete requests do not enter rendering after the current
preparation completes and cannot replace or republish the active frame.
