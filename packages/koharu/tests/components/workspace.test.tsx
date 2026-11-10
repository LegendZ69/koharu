import { QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CanvasWorkspace } from '@/components/editor/CanvasWorkspace'
import { pageKey, pagesKey, projectKey, queryClient } from '@/lib/queries'
import { receiveJob, useKoharuStore } from '@/lib/store'
import { commands, type Job, type Layer } from '@koharu/bridge/protocol'
import { TooltipProvider } from '@koharu/ui/components/tooltip'

const canvas = vi.hoisted(() => ({
  resize: vi.fn(),
  setView: vi.fn(),
  stageManifest: vi.fn(),
  installResource: vi.fn(),
  activateFrame: vi.fn(),
  activatePage: vi.fn(),
  clear: vi.fn(),
  previewOpacity: vi.fn(),
  beginTransform: vi.fn(),
  updateTransform: vi.fn(),
  finishTransform: vi.fn(),
  cancelTransform: vi.fn(),
  beginStroke: vi.fn(),
  extendStroke: vi.fn(),
  finishStroke: vi.fn(),
  cancelStroke: vi.fn(),
  sampleColor: vi.fn(),
  dispose: vi.fn(),
}))

const canvasState = vi.hoisted(() => ({
  canvas,
  error: null as Error | null,
  generation: 1 as number | null,
  hasFrame: true,
  retry: vi.fn(),
  status: 'ready' as 'loading' | 'switching' | 'ready' | 'recovering' | 'error',
}))
// The workspace's presenter of staged frames, as the mocked useCanvas last received it.
const presenter = vi.hoisted(() => ({
  present: (_generation: number, activate: () => boolean) => activate(),
}))
const prefetchCanvasPages = vi.hoisted(() => vi.fn(async () => []))

vi.mock('@/components/editor/useCanvas', () => ({
  useCanvas: (
    _element: HTMLCanvasElement | null,
    _revision: number | null,
    _generation: number,
    present: typeof presenter.present,
  ) => {
    presenter.present = present
    return canvasState
  },
}))
vi.mock('@koharu/bridge/canvas', () => ({
  prefetchCanvasPages,
  workspaceColor: () => [245, 245, 245],
}))

const layer: Layer = {
  type: 'image',
  id: 'element',
  parent: 'page',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  visibility: { visible: true, opacity: 1 },
  image: 'image',
}

const paintLayer: Layer = {
  type: 'raster',
  id: 'paint',
  parent: 'page',
  visibility: { visible: true, opacity: 1 },
  image: 'paint-image',
  name: 'Paint 1',
  kind: 'paint',
}

const inpaintJob: Job = {
  id: 'job',
  state: 'running',
  completed: 0,
  total: 1,
  page: 'page',
  stage: 'inpainting',
  model: null,
  error: null,
}

let nextAnimationFrame = 1
let animationFrames = new Map<number, FrameRequestCallback>()

beforeEach(() => {
  nextAnimationFrame = 1
  animationFrames = new Map()
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const frame = nextAnimationFrame++
    animationFrames.set(frame, callback)
    return frame
  })
  vi.stubGlobal('cancelAnimationFrame', (frame: number) => animationFrames.delete(frame))
  canvasState.error = null
  canvasState.generation = 1
  canvasState.hasFrame = true
  canvasState.status = 'ready'
  prefetchCanvasPages.mockClear()
})

afterEach(() => vi.unstubAllGlobals())

function installProject() {
  const page = {
    id: 'page',
    label: 'Page',
    size: { width: 1000, height: 1000 },
    layers: [layer],
    regions: [],
  }
  queryClient.setQueryData(projectKey, {
    name: 'Book',
    revision: 1,
    active_page: 'page',
    can_undo: false,
    can_redo: false,
  })
  queryClient.setQueryData(pagesKey, [])
  queryClient.setQueryData(pageKey, page)
  useKoharuStore.setState({ selectedLayers: [], tool: 'select' })
  useKoharuStore.setState({
    canvasPage: 'page',
    canvasRevision: 1,
    canvasGeneration: 1,
    canvasSize: [1000, 1000],
  })
}

function renderWorkspace() {
  render(
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <CanvasWorkspace />
      </TooltipProvider>
    </QueryClientProvider>,
  )
  const surface = screen.getByLabelText('Koharu canvas')
  Object.defineProperty(surface, 'getBoundingClientRect', {
    value: () => ({ x: 10, y: 20, width: 800, height: 600 }),
  })
  return surface
}

describe('canvas interaction adapter', () => {
  it('prefetches only after the authoritative canvas page and generation are active', async () => {
    installProject()
    queryClient.setQueryData(pagesKey, [
      {
        id: 'page',
        label: 'Page',
        size: { width: 1000, height: 1000 },
        source_asset: null,
        layer_count: 1,
      },
      {
        id: 'next',
        label: 'Next',
        size: { width: 1000, height: 1000 },
        source_asset: null,
        layer_count: 1,
      },
    ])
    useKoharuStore.setState({ canvasPage: 'previous' })
    renderWorkspace()

    await Promise.resolve()
    expect(prefetchCanvasPages).not.toHaveBeenCalled()

    act(() => useKoharuStore.setState({ canvasPage: 'page' }))
    await waitFor(() => expect(prefetchCanvasPages).toHaveBeenCalledWith(['next']))
  })

  it('renders an accessible browser canvas and keeps camera updates local', async () => {
    installProject()
    const surface = renderWorkspace()
    expect(screen.getByTestId('webgpu-canvas')).toBeInstanceOf(HTMLCanvasElement)

    fireEvent.wheel(surface, { clientX: 100, clientY: 100, deltaY: 4 })
    fireEvent.wheel(surface, { clientX: 100, clientY: 100, deltaY: 6 })

    await waitFor(() => expect(canvas.setView).toHaveBeenLastCalledWith(1, [0, -10]))
  })

  it('preserves a manual camera when a newer canvas generation becomes ready', () => {
    installProject()
    renderWorkspace()
    const camera = { zoom: 2, translation: [-120, -80] as [number, number], fit: null }

    act(() => useKoharuStore.setState({ camera }))
    act(() => {
      canvasState.status = 'switching'
      useKoharuStore.setState({ canvasGeneration: 2 })
    })
    const activate = vi.fn(() => true)
    act(() => void presenter.present(2, activate))
    act(() => {
      canvasState.status = 'ready'
      canvasState.generation = 2
      useKoharuStore.setState({ canvasRevision: 2 })
    })

    expect(activate).toHaveBeenCalledOnce()
    expect(useKoharuStore.getState().camera).toEqual(camera)
  })

  it('scrolls a width fit vertically, refits it on the next page, and drops it on zoom', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { size: { width: number; height: number } }) => ({
      ...page,
      size: { width: 1000, height: 2000 },
    }))
    useKoharuStore.setState({ canvasSize: [1000, 2000] })
    const surface = renderWorkspace()
    fireEvent(window, new Event('resize'))
    const camera = () => useKoharuStore.getState().camera

    // 800 CSS px at dpr 1 leaves 720 px between the insets; the tall page starts at its top.
    fireEvent.keyDown(window, { key: 'w' })
    expect(camera()).toEqual({ zoom: 0.72, translation: [64, 16], fit: 'width' })

    fireEvent.wheel(surface, { deltaX: 30, deltaY: 100 })
    expect(camera()).toEqual({ zoom: 0.72, translation: [64, -84], fit: 'width' })
    fireEvent.wheel(surface, { deltaY: -500 })
    expect(camera().translation).toEqual([64, 16])

    act(() => useKoharuStore.setState({ tool: 'pan' }))
    fireEvent.pointerDown(surface, { button: 0, pointerId: 15, clientX: 100, clientY: 300 })
    fireEvent.pointerMove(surface, { pointerId: 15, clientX: 300, clientY: 250 })
    fireEvent.pointerUp(surface, { pointerId: 15, clientX: 300, clientY: 250 })
    expect(camera()).toEqual({ zoom: 0.72, translation: [64, -34], fit: 'width' })

    act(() => {
      canvasState.generation = 2
      useKoharuStore.setState({ canvasGeneration: 2, canvasRevision: 2 })
    })
    expect(camera()).toEqual({ zoom: 0.72, translation: [64, -34], fit: 'width' })

    // An uncached page reaches the query and the native canvas before its frame is presented.
    await act(async () => {
      queryClient.setQueryData(pageKey, {
        id: 'next',
        label: 'Next',
        size: { width: 500, height: 2000 },
        layers: [],
        regions: [],
      })
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    act(() =>
      useKoharuStore.setState({ canvasPage: 'next', canvasGeneration: 3, canvasSize: [500, 2000] }),
    )
    expect(camera()).toEqual({ zoom: 0.72, translation: [64, -34], fit: 'width' })

    // Its first frame is activated with its own fitted view on the canvas. A frame that is not
    // activated hands the camera back to the page still shown.
    const views: unknown[] = []
    const activation = (activated: boolean) => () => {
      views.push(canvas.setView.mock.lastCall)
      return activated
    }
    act(() => void presenter.present(3, activation(false)))
    expect(camera()).toEqual({ zoom: 0.72, translation: [64, -34], fit: 'width' })
    expect(canvas.setView).toHaveBeenLastCalledWith(0.72, [64, -34])
    act(() => void presenter.present(3, activation(true)))
    expect(views).toEqual([
      [1.44, [64, 16]],
      [1.44, [64, 16]],
    ])
    expect(camera()).toEqual({ zoom: 1.44, translation: [64, 16], fit: 'width' })

    // A newer generation of that page can be presented before the workspace sees the first one on
    // screen. The page keeps its camera, scroll included.
    fireEvent.wheel(surface, { deltaY: 100 })
    act(() => useKoharuStore.setState({ canvasGeneration: 4 }))
    act(() => void presenter.present(4, () => true))
    act(() => {
      canvasState.generation = 4
      useKoharuStore.setState({ canvasRevision: 4 })
    })
    expect(camera()).toEqual({ zoom: 1.44, translation: [64, -84], fit: 'width' })

    fireEvent.wheel(surface, { clientX: 100, clientY: 100, ctrlKey: true, deltaY: -100 })
    expect(camera().fit).toBeNull()
    expect(useKoharuStore.getState().fitMode).toBe('width')
  })

  it('suppresses Alt browser handling while an editor input retains focus', () => {
    installProject()
    renderWorkspace()
    const input = document.createElement('input')
    document.body.append(input)
    input.focus()

    const event = new KeyboardEvent('keydown', { key: 'Alt', bubbles: true, cancelable: true })
    input.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
    input.remove()
  })

  it('announces WebGPU startup failures and offers recovery', () => {
    installProject()
    canvasState.status = 'error'
    canvasState.error = new Error('No compatible WebGPU adapter was found.')
    renderWorkspace()
    expect(screen.getByRole('alert')).toHaveTextContent('WebGPU canvas unavailable')
    expect(screen.getByRole('alert')).toHaveTextContent('No compatible WebGPU adapter was found.')
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(canvasState.retry).toHaveBeenCalledOnce()
  })

  it('previews brush input locally and sends only the durable paint commit to Rust', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw', brush: { diameter: 48, color: '#FFFFFF' } })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()
    expect(surface).toHaveStyle({ cursor: 'none' })

    fireEvent.pointerDown(surface, { button: 0, pointerId: 7, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 7, clientX: 55, clientY: 65 })
    fireEvent.pointerUp(surface, { pointerId: 7, clientX: 58, clientY: 70 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'paint',
      layer: null,
      point: { x: 20, y: 20 },
      diameter: 48,
      color: [255, 255, 255, 255],
    })
    expect(canvas.extendStroke).toHaveBeenCalledWith(expect.arrayContaining([{ x: 45, y: 45 }]))
    expect(canvas.finishStroke).toHaveBeenCalledOnce()
    expect(commit).toHaveBeenCalledWith(1, null, expect.arrayContaining([{ x: 45, y: 45 }]), {
      diameter: 48,
      color: [255, 255, 255, 255],
    })
  })

  it.each([
    ['revision', { canvasRevision: 2 }],
    ['generation', { canvasGeneration: 2 }],
  ])('cancels an active gesture when the canvas %s changes', async (_name, update) => {
    installProject()
    useKoharuStore.setState({ tool: 'draw', brush: { diameter: 48, color: '#FFFFFF' } })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 8, clientX: 30, clientY: 40 })
    expect(canvas.beginStroke).toHaveBeenCalledOnce()

    act(() => {
      useKoharuStore.setState(update)
    })

    await waitFor(() => expect(canvas.cancelStroke).toHaveBeenCalledOnce())
    fireEvent.pointerUp(surface, { pointerId: 8, clientX: 30, clientY: 40 })
    expect(canvas.finishStroke).not.toHaveBeenCalled()
    expect(commit).not.toHaveBeenCalled()
  })

  it('uses rendered text bounds for hit testing and semantic transforms', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'element',
          parent: 'page',
          geometry: layer.geometry,
          angle_degrees: 0,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'en' },
            translation: { text: 'Rendered', language: null },
            role: null,
            source_region: null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: null,
        },
      ],
    }))
    useKoharuStore.setState({
      layerFrames: {
        element: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    })
    const commit = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 9, clientX: 50, clientY: 60 })
    fireEvent.pointerMove(surface, { pointerId: 9, clientX: 70, clientY: 80 })
    fireEvent.pointerUp(surface, { pointerId: 9, clientX: 70, clientY: 80 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element'])
    expect(canvas.beginTransform).toHaveBeenCalledWith([
      {
        element: 'element',
        frame: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    ])
    expect(canvas.updateTransform).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          element: 'element',
          frame: expect.objectContaining({ x: 50, y: 60 }),
        }),
      ]),
    )
    expect(canvas.finishTransform).toHaveBeenCalledOnce()
    expect(commit).toHaveBeenCalledWith(
      1,
      expect.arrayContaining([
        expect.objectContaining({
          element: 'element',
          frame: expect.objectContaining({ x: 50, y: 60 }),
        }),
      ]),
    )
  })

  it('resizes a selected layer through Koharu selection controls', async () => {
    installProject()
    useKoharuStore.setState({ selectedLayers: ['element'] })
    const commit = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
    renderWorkspace()
    Object.defineProperty(screen.getByTestId('canvas-overlay'), 'getBoundingClientRect', {
      value: () => ({ x: 10, y: 20, width: 800, height: 600 }),
    })
    const handle = document.querySelector<HTMLElement>('[data-resize-handle="e"]')!

    fireEvent.pointerDown(handle, { button: 0, pointerId: 10, clientX: 120, clientY: 65 })
    fireEvent.pointerMove(handle, { pointerId: 10, clientX: 140, clientY: 65 })
    fireEvent.pointerUp(handle, { pointerId: 10, clientX: 140, clientY: 65 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginTransform).toHaveBeenCalledWith([
      { element: 'element', frame: { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 } },
    ])
    expect(canvas.updateTransform).toHaveBeenCalledWith(
      expect.arrayContaining([
        {
          element: 'element',
          frame: { x: 10, y: 20, width: 120, height: 50, angle_degrees: 0 },
        },
      ]),
    )
    expect(commit).toHaveBeenCalledWith(
      1,
      expect.arrayContaining([
        {
          element: 'element',
          frame: { x: 10, y: 20, width: 120, height: 50, angle_degrees: 0 },
        },
      ]),
    )
  })

  it('shows the automatic region behind the selected text controls', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'element',
          parent: 'page',
          geometry: null,
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'en' },
            translation: { text: 'Rendered', language: null },
            role: null,
            source_region: null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: 'bubble',
        },
      ],
      regions: [
        {
          id: 'bubble',
          parent: 'page',
          geometry: {
            points: [
              { x: 20, y: 30 },
              { x: 100, y: 30 },
              { x: 100, y: 90 },
              { x: 20, y: 90 },
            ],
          },
          kind: 'bubble',
          label: null,
        },
      ],
    }))
    useKoharuStore.setState({
      selectedLayers: ['element'],
      layerFrames: {
        element: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    })

    renderWorkspace()

    expect(screen.getByTestId('text-fit-region').querySelector('polygon')).toHaveAttribute(
      'points',
      '20,30 100,30 100,90 20,90',
    )
  })

  it('targets the selected raster layer with the eraser', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, paintLayer],
    }))
    useKoharuStore.setState({ tool: 'eraser', selectedLayers: ['paint'] })
    const commit = vi
      .spyOn(commands, 'commitErase')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 11, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 11, clientX: 30, clientY: 40 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'erase',
      layer: 'paint',
      point: { x: 20, y: 20 },
      diameter: 48,
    })
    expect(commit).toHaveBeenCalledWith(1, 'paint', expect.arrayContaining([{ x: 20, y: 20 }]), 48)
  })

  it('maps the Remove tool to an inpainting mask gesture', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'remove' })
    const commit = vi.spyOn(commands, 'commitInpaint').mockResolvedValue('job')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 12, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 12, clientX: 30, clientY: 40 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'inpaint',
      layer: null,
      point: { x: 20, y: 20 },
      diameter: 48,
    })
    expect(commit).toHaveBeenCalledWith(1, expect.arrayContaining([{ x: 20, y: 20 }]), 48)
  })

  it.each([
    ['fails', 'failed'],
    ['is stopped', 'stopped'],
  ] as const)('releases a pending Remove stroke when its job %s', async (_name, state) => {
    installProject()
    useKoharuStore.setState({ tool: 'remove' })
    const commit = vi.spyOn(commands, 'commitInpaint').mockResolvedValue('job')
    receiveJob(inpaintJob)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 16, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 16, clientX: 30, clientY: 40 })
    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    await act(async () => undefined)
    expect(canvas.cancelStroke).not.toHaveBeenCalled()

    act(() => receiveJob({ ...inpaintJob, state }))
    expect(canvas.cancelStroke).toHaveBeenCalledOnce()
  })

  it.each([
    ['fails', 'failed'],
    ['is stopped', 'stopped'],
  ] as const)(
    'keeps a Remove stroke pending while its job is unreported, until the job %s',
    async (_name, state) => {
      installProject()
      useKoharuStore.setState({ tool: 'remove' })
      const commit = vi.spyOn(commands, 'commitInpaint').mockResolvedValue('job')
      const surface = renderWorkspace()

      fireEvent.pointerDown(surface, { button: 0, pointerId: 18, clientX: 30, clientY: 40 })
      fireEvent.pointerUp(surface, { pointerId: 18, clientX: 30, clientY: 40 })
      await waitFor(() => expect(commit).toHaveBeenCalledOnce())
      await act(async () => undefined)
      act(() => receiveJob(inpaintJob))
      expect(canvas.cancelStroke).not.toHaveBeenCalled()

      act(() => receiveJob({ ...inpaintJob, state }))
      expect(canvas.cancelStroke).toHaveBeenCalledOnce()
    },
  )

  it('releases a Remove stroke whose job ended before its commit returned', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'remove' })
    const result = Promise.withResolvers<string | null>()
    const commit = vi.spyOn(commands, 'commitInpaint').mockReturnValue(result.promise)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 19, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 19, clientX: 30, clientY: 40 })
    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    act(() => {
      receiveJob(inpaintJob)
      receiveJob({ ...inpaintJob, state: 'stopped' })
    })
    expect(canvas.cancelStroke).not.toHaveBeenCalled()

    await act(async () => result.resolve('job'))
    await waitFor(() => expect(canvas.cancelStroke).toHaveBeenCalledOnce())
  })

  it('leaves a pending Remove stroke to the newer revision its job commits', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'remove' })
    const commit = vi.spyOn(commands, 'commitInpaint').mockResolvedValue('job')
    receiveJob(inpaintJob)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 17, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 17, clientX: 30, clientY: 40 })
    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    await act(async () => undefined)

    act(() => useKoharuStore.setState({ canvasRevision: 2, canvasGeneration: 2 }))
    act(() => receiveJob({ ...inpaintJob, state: 'finished', completed: 1 }))
    expect(canvas.cancelStroke).not.toHaveBeenCalled()
  })

  it('creates point text on click and paragraph text on drag', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'text' })
    const point = vi
      .spyOn(commands, 'addPointText')
      .mockResolvedValue({ revision: 2, layer: 'point-text' })
    const box = vi
      .spyOn(commands, 'addTextBox')
      .mockResolvedValue({ revision: 3, layer: 'box-text' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 13, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 13, clientX: 30, clientY: 40 })
    await waitFor(() => expect(point).toHaveBeenCalledWith({ x: 20, y: 20 }))

    fireEvent.pointerDown(surface, { button: 0, pointerId: 14, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    await waitFor(() => expect(box).toHaveBeenCalledOnce())
    expect(box).toHaveBeenCalledWith({
      x: 30,
      y: 30,
      width: 100,
      height: 60,
      angle_degrees: 0,
    })
  })
})
